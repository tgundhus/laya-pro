"""Bundle the canonical inference modules into a separate, non-overlapping namespace."""
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


HERE = Path(__file__).resolve().parent
MODULES = (
    "onnx_agent", "inference_common", "inference", "checkpoint", "calibration",
    "hooks", "revisions", "confidence", "performance", "structured", "consistency",
)


def shared_source():
    # An sdist carries its own inputs, so rebuilding it never needs the original checkout.
    source = HERE / "_shared"
    return source if source.is_dir() else HERE.parents[1] / "laya"


def copy_shared(command, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for module in MODULES:
        source = shared_source() / (module + ".py")
        if not source.is_file():
            raise FileNotFoundError("missing canonical runtime source: %s" % source)
        command.copy_file(str(source), str(destination / source.name))


class BuildRuntime(build_py):
    def run(self):
        super().run()
        copy_shared(self, Path(self.build_lib) / "laya_onnx" / "_runtime")

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        return outputs + [str(Path(self.build_lib) / "laya_onnx" / "_runtime" / (name + ".py"))
                          for name in MODULES]


class SourceRuntime(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        copy_shared(self, Path(base_dir) / "_shared")


setup(cmdclass={"build_py": BuildRuntime, "sdist": SourceRuntime})
