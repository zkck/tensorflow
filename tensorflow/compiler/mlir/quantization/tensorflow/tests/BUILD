# TODO(b/)220688154): Enable MLIR test cases.
#load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    default_visibility = [
        "//tensorflow/compiler/mlir/quantization:__subpackages__",
    ],
    licenses = ["notice"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir/quantization/tensorflow:tf-quant-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
        "@llvm-project//mlir:run_lit.sh",
    ],
)

# TODO(b/)220688154): Enable MLIR test cases.
#glob_lit_tests(
#    data = [":test_utilities"],
#    driver = "@llvm-project//mlir:run_lit.sh",
#    test_file_exts = ["mlir"],
#)
