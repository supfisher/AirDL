# -*- Mode: python; py-indent-offset: 4; indent-tabs-mode: nil; coding: utf-8; -*-

# def options(opt):
#     pass

# def configure(conf):
#     conf.check_nonfatal(header_name='stdint.h', define_name='HAVE_STDINT_H')
    # conf.env['ladd'] = conf.check(mandatory=True, lib='add', uselib_store='LADD')
    # conf.env.append_value("LINKFLAGS", ["-L/home/mag0a/mount/Projects/FLinMEN/ns3/bake/source/test/", "-ladd"])
    # conf.load('compiler_cxx')
    # conf.env.append_value('CXXFLAGS', ['-D_GLIBCXX_USE_CXX11_ABI=0', '-std=gnu++14'])
    # conf.env.append_value('INCLUDES', ['/home/mag0a/mount/Projects/FLinMEN/ns3/bake/source/torchC/libtorch/include', '/home/mag0a/mount/Projects/FLinMEN/ns3/bake/source/torchC/libtorch/include/torch/csrc/api/include'])
    # conf.env.append_value('LIBPATH', ['/home/mag0a/mount/Projects/FLinMEN/ns3/bake/source/torchC/libtorch/lib/'])
    # conf.env.append_value('RPATH', ['/home/mag0a/mount/Projects/FLinMEN/ns3/bake/source/torchC/libtorch/lib/'])
    # conf.env.append_value('LIB', ['c10', 'torch'])

    # conf.env.append_value('INCLUDES', ['/home/mag0a/mount/Projects/FLinMEN/ns3/bake/source/test/'])
    # conf.env.append_value('LIBPATH', ['/home/mag0a/mount/Projects/FLinMEN/ns3/bake/source/test/'])
    # conf.env.append_value('LIB', ['add'])


def build(bld):

    # module = bld.create_ns3_module('distributedml', ['mpi', 'applications', 'internet', 'config-store','stats'])
    module = bld.create_ns3_module('distributedml', ['mpi', 'applications', 'wifi', 'energy'])

    module.source = [
        'model/distributed-ml-utils.cc',
        'helper/distributed-ml-tcp-helper.cc',
        'model/distributed-ml-mpi.cc',
        'model/distributed-ml-agent.cc',
        'model/distributed-ml-traces.cc'
        ]

    module_test = bld.create_ns3_module_test_library('distributedml')
    module_test.source = [
        # 'test/distributedml-test-suite.cc',
        ]
    # Tests encapsulating example programs should be listed here
    if (bld.env['ENABLE_EXAMPLES']):
        module_test.source.extend([
        #    'test/distributed-ml-test-examples-test-suite.cc',
             ])

    headers = bld(features='ns3header')
    headers.module = 'distributedml'
    headers.source = [
		'model/distributed-ml-utils.h',
        'helper/distributed-ml-tcp-helper.h',
        'model/distributed-ml-mpi.h',
        'model/distributed-ml-agent.h',
        'model/distributed-ml-traces.h'
        ]

    if bld.env.ENABLE_EXAMPLES:
        bld.recurse('examples')


    bld.ns3_python_bindings()

