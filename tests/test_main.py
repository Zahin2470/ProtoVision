"""
Unit tests for main.py (the CLI).

Argument parsing is tested directly against build_parser(). Command
dispatch (cmd_enroll/cmd_live/cmd_list) is tested by monkeypatching
load_default_backbone/EnrollApp/LiveApp with fakes, so we verify the CLI
wires arguments through correctly and handles both success and
error/cancellation paths — without needing a camera or real DINOv3 weights.
cmd_list needs neither a camera nor a backbone, so it's tested directly,
no mocking required.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
from protovision.backbone import DinoV3NotAvailableError
from protovision.enroll import EnrollState
from protovision.prototypes import PrototypeStore


# --------------------------------------------------------------------------
# fakes for cmd_enroll / cmd_live dispatch tests
# --------------------------------------------------------------------------

def make_fake_enroll_app_cls(run_return, captured_count=None):
    """Fresh fake EnrollApp class per call — avoids cross-test state leakage.
    `.created` records every instance so tests can inspect what kwargs
    main.py actually passed through."""
    created = []

    class _FakeEnrollApp:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.label = kwargs["label"]
            self._captured = captured_count if captured_count is not None else kwargs.get("target_examples", 0)
            created.append(self)

        def run(self):
            return run_return

        @property
        def progress(self):
            return (self._captured, self.kwargs.get("target_examples"))

    _FakeEnrollApp.created = created
    return _FakeEnrollApp


def make_fake_live_app_cls():
    created = []

    class _FakeLiveApp:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.ran = False
            created.append(self)

        def run(self):
            self.ran = True

    _FakeLiveApp.created = created
    return _FakeLiveApp


def make_fake_audio_manager_cls():
    """Fresh fake AudioManager class per call — records construction kwargs
    (so tests can check --mute was respected) and start/stop_ambient calls
    (so tests can check cmd_live's ambient-audio lifecycle) without
    touching real pygame."""
    created = []

    class _FakeAudioManager:
        def __init__(self, enabled=True):
            self.enabled = enabled
            self.start_ambient_calls = 0
            self.stop_ambient_calls = 0
            created.append(self)

        def start_ambient(self):
            self.start_ambient_calls += 1
            return True

        def stop_ambient(self):
            self.stop_ambient_calls += 1

    _FakeAudioManager.created = created
    return _FakeAudioManager


def base_enroll_args(**overrides):
    defaults = dict(
        label="mug", store="unused.json", repo=None, weights=None, device="cpu", mute=False,
        target_examples=8, min_examples=5, box_fraction=0.5,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def base_live_args(**overrides):
    defaults = dict(
        store="unused.json", repo=None, weights=None, device="cpu", mute=False,
        threshold=0.5, match_mode="mean", frame_skip=5, box_fraction=0.5,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------

class TestArgumentParsing:
    def test_enroll_requires_label(self):
        parser = main.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["enroll"])

    def test_enroll_defaults(self):
        parser = main.build_parser()
        args = parser.parse_args(["enroll", "--label", "mug"])
        assert args.target_examples == 8
        assert args.min_examples == 5
        assert args.box_fraction == 0.5
        assert args.store == main.DEFAULT_STORE_PATH
        assert args.device == "cpu"
        assert args.mute is False

    def test_mute_flag_defaults_to_false(self):
        parser = main.build_parser()
        args = parser.parse_args(["live"])
        assert args.mute is False

    def test_mute_flag_can_be_set(self):
        parser = main.build_parser()
        args = parser.parse_args(["live", "--mute"])
        assert args.mute is True

    def test_mute_flag_works_after_subcommand_for_enroll_too(self):
        parser = main.build_parser()
        args = parser.parse_args(["enroll", "--label", "mug", "--mute"])
        assert args.mute is True

    def test_enroll_overrides(self):
        parser = main.build_parser()
        args = parser.parse_args(
            [
                "enroll", "--label", "mug",
                "--target-examples", "10", "--min-examples", "6",
                "--box-fraction", "0.7", "--store", "custom.json", "--device", "cuda",
            ]
        )
        assert args.label == "mug"
        assert args.target_examples == 10
        assert args.min_examples == 6
        assert args.box_fraction == 0.7
        assert args.store == "custom.json"
        assert args.device == "cuda"

    def test_live_defaults(self):
        parser = main.build_parser()
        args = parser.parse_args(["live"])
        assert args.threshold == 0.5
        assert args.match_mode == "mean"
        assert args.frame_skip == 5
        assert args.box_fraction == 0.5

    def test_live_overrides(self):
        parser = main.build_parser()
        args = parser.parse_args(
            ["live", "--threshold", "0.6", "--match-mode", "max", "--frame-skip", "3"]
        )
        assert args.threshold == 0.6
        assert args.match_mode == "max"
        assert args.frame_skip == 3

    def test_live_rejects_invalid_match_mode(self):
        parser = main.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["live", "--match-mode", "bogus"])

    def test_rejects_invalid_device(self):
        parser = main.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["enroll", "--label", "mug", "--device", "bogus"])

    def test_list_command_parses(self):
        parser = main.build_parser()
        args = parser.parse_args(["list"])
        assert args.command == "list"

    def test_missing_command_exits(self):
        parser = main.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_shared_flags_work_after_subcommand(self):
        # Confirms the parents=[common] wiring — shared flags aren't only
        # valid before the subcommand name.
        parser = main.build_parser()
        args = parser.parse_args(["enroll", "--label", "mug", "--store", "x.json", "--device", "cpu"])
        assert args.store == "x.json"

    def test_unknown_command_exits(self):
        parser = main.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["bogus-command"])


# --------------------------------------------------------------------------
# cmd_list — no mocking needed, no camera/backbone involved at all
# --------------------------------------------------------------------------

class TestCmdList:
    def test_empty_store_prints_message(self, tmp_path, capsys):
        args = argparse.Namespace(store=str(tmp_path / "missing.json"))
        rc = main.cmd_list(args)
        assert rc == 0
        assert "No classes enrolled" in capsys.readouterr().out

    def test_lists_classes_and_counts(self, tmp_path, capsys):
        store = PrototypeStore()
        store.add_example("mug", np.zeros(4, dtype=np.float32))
        store.add_example("mug", np.ones(4, dtype=np.float32))
        store.add_example("bottle", np.ones(4, dtype=np.float32))
        store_path = tmp_path / "p.json"
        store.save(store_path)

        args = argparse.Namespace(store=str(store_path))
        rc = main.cmd_list(args)
        out = capsys.readouterr().out

        assert rc == 0
        assert "mug: 2 example(s)" in out
        assert "bottle: 1 example(s)" in out

    def test_lists_classes_alphabetically(self, tmp_path, capsys):
        store = PrototypeStore()
        store.add_example("zebra", np.zeros(4, dtype=np.float32))
        store.add_example("apple", np.zeros(4, dtype=np.float32))
        store_path = tmp_path / "p.json"
        store.save(store_path)

        args = argparse.Namespace(store=str(store_path))
        main.cmd_list(args)
        out = capsys.readouterr().out
        assert out.index("apple") < out.index("zebra")


# --------------------------------------------------------------------------
# _load_backbone_or_exit
# --------------------------------------------------------------------------

class TestLoadBackboneOrExit:
    def test_returns_backbone_on_success(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(main, "load_default_backbone", lambda **kwargs: sentinel)
        args = argparse.Namespace(repo=None, weights=None, device="cpu")
        assert main._load_backbone_or_exit(args) is sentinel

    def test_exits_with_code_1_when_unavailable(self, monkeypatch, capsys):
        def raiser(**kwargs):
            raise DinoV3NotAvailableError("weights not found")

        monkeypatch.setattr(main, "load_default_backbone", raiser)
        args = argparse.Namespace(repo=None, weights=None, device="cpu")
        with pytest.raises(SystemExit) as exc_info:
            main._load_backbone_or_exit(args)
        assert exc_info.value.code == 1
        assert "weights not found" in capsys.readouterr().err


# --------------------------------------------------------------------------
# cmd_enroll dispatch
# --------------------------------------------------------------------------

class TestCmdEnroll:
    def test_success_path_returns_zero_and_saves_message(self, monkeypatch, capsys):
        monkeypatch.setattr(main, "load_default_backbone", lambda **kwargs: object())
        fake_cls = make_fake_enroll_app_cls(EnrollState.DONE, captured_count=5)
        monkeypatch.setattr(main, "EnrollApp", fake_cls)

        rc = main.cmd_enroll(base_enroll_args(target_examples=5))

        assert rc == 0
        assert "Saved 5 example" in capsys.readouterr().out

    def test_cancelled_path_returns_one(self, monkeypatch, capsys):
        monkeypatch.setattr(main, "load_default_backbone", lambda **kwargs: object())
        fake_cls = make_fake_enroll_app_cls(EnrollState.CANCELLED)
        monkeypatch.setattr(main, "EnrollApp", fake_cls)

        rc = main.cmd_enroll(base_enroll_args())

        assert rc == 1
        assert "cancelled" in capsys.readouterr().out.lower()

    def test_passes_label_and_options_through_to_enroll_app(self, monkeypatch):
        monkeypatch.setattr(main, "load_default_backbone", lambda **kwargs: object())
        fake_cls = make_fake_enroll_app_cls(EnrollState.DONE, captured_count=1)
        monkeypatch.setattr(main, "EnrollApp", fake_cls)

        main.cmd_enroll(base_enroll_args(label="bottle", target_examples=6, min_examples=4, box_fraction=0.4))

        passed = fake_cls.created[0].kwargs
        assert passed["label"] == "bottle"
        assert passed["target_examples"] == 6
        assert passed["min_examples"] == 4
        assert passed["box_fraction"] == 0.4

    def test_exits_when_backbone_unavailable(self, monkeypatch):
        def raiser(**kwargs):
            raise DinoV3NotAvailableError("no repo")

        monkeypatch.setattr(main, "load_default_backbone", raiser)
        with pytest.raises(SystemExit):
            main.cmd_enroll(base_enroll_args())

    def test_audio_manager_passed_to_enroll_app(self, monkeypatch):
        monkeypatch.setattr(main, "load_default_backbone", lambda **kwargs: object())
        fake_cls = make_fake_enroll_app_cls(EnrollState.DONE, captured_count=1)
        monkeypatch.setattr(main, "EnrollApp", fake_cls)
        fake_audio_cls = make_fake_audio_manager_cls()
        monkeypatch.setattr(main, "AudioManager", fake_audio_cls)

        main.cmd_enroll(base_enroll_args())

        audio_instance = fake_audio_cls.created[0]
        passed = fake_cls.created[0].kwargs
        assert passed["audio"] is audio_instance

    def test_mute_flag_disables_audio_manager(self, monkeypatch):
        monkeypatch.setattr(main, "load_default_backbone", lambda **kwargs: object())
        monkeypatch.setattr(main, "EnrollApp", make_fake_enroll_app_cls(EnrollState.DONE, captured_count=1))
        fake_audio_cls = make_fake_audio_manager_cls()
        monkeypatch.setattr(main, "AudioManager", fake_audio_cls)

        main.cmd_enroll(base_enroll_args(mute=True))

        assert fake_audio_cls.created[0].enabled is False

    def test_unmuted_by_default(self, monkeypatch):
        monkeypatch.setattr(main, "load_default_backbone", lambda **kwargs: object())
        monkeypatch.setattr(main, "EnrollApp", make_fake_enroll_app_cls(EnrollState.DONE, captured_count=1))
        fake_audio_cls = make_fake_audio_manager_cls()
        monkeypatch.setattr(main, "AudioManager", fake_audio_cls)

        main.cmd_enroll(base_enroll_args(mute=False))

        assert fake_audio_cls.created[0].enabled is True


# --------------------------------------------------------------------------
# cmd_live dispatch
# --------------------------------------------------------------------------

class TestCmdLive:
    def test_no_warning_when_store_has_classes(self, monkeypatch, tmp_path, capsys):
        store = PrototypeStore()
        store.add_example("mug", np.zeros(4, dtype=np.float32))
        store_path = tmp_path / "p.json"
        store.save(store_path)

        monkeypatch.setattr(main, "load_default_backbone", lambda **kwargs: object())
        monkeypatch.setattr(main, "LiveApp", make_fake_live_app_cls())

        rc = main.cmd_live(base_live_args(store=str(store_path)))

        assert rc == 0
        assert "warning" not in capsys.readouterr().err.lower()

    def test_warns_when_store_is_empty(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(main, "load_default_backbone", lambda **kwargs: object())
        monkeypatch.setattr(main, "LiveApp", make_fake_live_app_cls())

        rc = main.cmd_live(base_live_args(store=str(tmp_path / "missing.json")))

        assert rc == 0
        assert "no classes enrolled" in capsys.readouterr().err.lower()

    def test_passes_options_through_to_live_app(self, monkeypatch, tmp_path):
        store_path = tmp_path / "p.json"
        PrototypeStore().save(store_path)  # exists but empty, avoids the warning noise here

        monkeypatch.setattr(main, "load_default_backbone", lambda **kwargs: object())
        fake_cls = make_fake_live_app_cls()
        monkeypatch.setattr(main, "LiveApp", fake_cls)

        main.cmd_live(base_live_args(store=str(store_path), threshold=0.7, match_mode="max", frame_skip=2))

        passed = fake_cls.created[0].kwargs
        assert passed["threshold"] == 0.7
        assert passed["match_mode"] == "max"
        assert passed["frame_skip"] == 2

    def test_calls_run_on_the_app(self, monkeypatch, tmp_path):
        store_path = tmp_path / "p.json"
        PrototypeStore().save(store_path)

        monkeypatch.setattr(main, "load_default_backbone", lambda **kwargs: object())
        fake_cls = make_fake_live_app_cls()
        monkeypatch.setattr(main, "LiveApp", fake_cls)

        main.cmd_live(base_live_args(store=str(store_path)))

        assert fake_cls.created[0].ran is True

    def test_exits_when_backbone_unavailable(self, monkeypatch, tmp_path):
        def raiser(**kwargs):
            raise DinoV3NotAvailableError("no weights")

        monkeypatch.setattr(main, "load_default_backbone", raiser)
        with pytest.raises(SystemExit):
            main.cmd_live(base_live_args(store=str(tmp_path / "p.json")))

    def test_audio_manager_passed_to_live_app(self, monkeypatch, tmp_path):
        store_path = tmp_path / "p.json"
        PrototypeStore().save(store_path)
        monkeypatch.setattr(main, "load_default_backbone", lambda **kwargs: object())
        fake_cls = make_fake_live_app_cls()
        monkeypatch.setattr(main, "LiveApp", fake_cls)
        fake_audio_cls = make_fake_audio_manager_cls()
        monkeypatch.setattr(main, "AudioManager", fake_audio_cls)

        main.cmd_live(base_live_args(store=str(store_path)))

        audio_instance = fake_audio_cls.created[0]
        passed = fake_cls.created[0].kwargs
        assert passed["audio"] is audio_instance

    def test_mute_flag_disables_audio_manager(self, monkeypatch, tmp_path):
        store_path = tmp_path / "p.json"
        PrototypeStore().save(store_path)
        monkeypatch.setattr(main, "load_default_backbone", lambda **kwargs: object())
        monkeypatch.setattr(main, "LiveApp", make_fake_live_app_cls())
        fake_audio_cls = make_fake_audio_manager_cls()
        monkeypatch.setattr(main, "AudioManager", fake_audio_cls)

        main.cmd_live(base_live_args(store=str(store_path), mute=True))

        assert fake_audio_cls.created[0].enabled is False

    def test_starts_and_stops_ambient_audio_around_run(self, monkeypatch, tmp_path):
        store_path = tmp_path / "p.json"
        PrototypeStore().save(store_path)
        monkeypatch.setattr(main, "load_default_backbone", lambda **kwargs: object())
        monkeypatch.setattr(main, "LiveApp", make_fake_live_app_cls())
        fake_audio_cls = make_fake_audio_manager_cls()
        monkeypatch.setattr(main, "AudioManager", fake_audio_cls)

        main.cmd_live(base_live_args(store=str(store_path)))

        audio_instance = fake_audio_cls.created[0]
        assert audio_instance.start_ambient_calls == 1
        assert audio_instance.stop_ambient_calls == 1

    def test_ambient_stopped_even_if_run_raises(self, monkeypatch, tmp_path):
        """The ambient loop shouldn't be left playing forever just because
        the camera loop crashed — stop_ambient() is in a finally block."""
        store_path = tmp_path / "p.json"
        PrototypeStore().save(store_path)
        monkeypatch.setattr(main, "load_default_backbone", lambda **kwargs: object())

        class _RaisingLiveApp:
            def __init__(self, **kwargs):
                pass

            def run(self):
                raise RuntimeError("simulated camera crash")

        monkeypatch.setattr(main, "LiveApp", _RaisingLiveApp)
        fake_audio_cls = make_fake_audio_manager_cls()
        monkeypatch.setattr(main, "AudioManager", fake_audio_cls)

        with pytest.raises(RuntimeError):
            main.cmd_live(base_live_args(store=str(store_path)))

        assert fake_audio_cls.created[0].stop_ambient_calls == 1


# --------------------------------------------------------------------------
# main() end-to-end dispatch
# --------------------------------------------------------------------------

class TestMainDispatch:
    def test_dispatches_list_command(self, tmp_path, capsys):
        rc = main.main(["list", "--store", str(tmp_path / "missing.json")])
        assert rc == 0
        assert "No classes enrolled" in capsys.readouterr().out

    def test_no_command_exits(self):
        with pytest.raises(SystemExit):
            main.main([])

    def test_enroll_command_exits_cleanly_when_backbone_unavailable(self, monkeypatch, tmp_path):
        def raiser(**kwargs):
            raise DinoV3NotAvailableError("no repo cloned")

        monkeypatch.setattr(main, "load_default_backbone", raiser)
        with pytest.raises(SystemExit) as exc:
            main.main(["enroll", "--label", "mug", "--store", str(tmp_path / "p.json")])
        assert exc.value.code == 1

    def test_live_command_exits_cleanly_when_backbone_unavailable(self, monkeypatch, tmp_path):
        def raiser(**kwargs):
            raise DinoV3NotAvailableError("no repo cloned")

        monkeypatch.setattr(main, "load_default_backbone", raiser)
        with pytest.raises(SystemExit) as exc:
            main.main(["live", "--store", str(tmp_path / "p.json")])
        assert exc.value.code == 1
