# -*- coding: utf-8 -*-

from numba import cuda

cuda.current_context().get_memory_info()
cuda.current_context().reset()