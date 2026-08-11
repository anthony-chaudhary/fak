def diamond_library(name, deps = []):
    native.filegroup(name = name, srcs = [name + ".txt"], data = deps, visibility = ["//visibility:public"])

def diamond_test(name, deps = []):
    native.genrule(name = name + "_test", srcs = deps, outs = [name + "_test.ok"], cmd = "echo test > $@")
