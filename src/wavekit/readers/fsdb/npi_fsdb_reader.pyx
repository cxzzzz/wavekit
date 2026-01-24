# cython: language_level=3
import math
import sys
import numpy as np
cimport numpy as np
#from cpython import PyBytes_FromString
import cython
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libc.string cimport strcpy, strlen
from .npi_fsdb cimport *

cdef class NpiFsdbScope:
    cdef npiFsdbScopeHandle scope_handle

    @staticmethod
    cdef init(npiFsdbScopeHandle scope_handle):
        scope = NpiFsdbScope()
        scope.scope_handle = scope_handle
        return scope

    def __init__(self):
        self.scope_handle = NULL

    def child_scope_list(self) -> list[NpiFsdbScope]:
        assert self.scope_handle != NULL

        res = []
        child_scope_handle_iter = npi_fsdb_iter_child_scope(self.scope_handle)
        cdef npiFsdbScopeHandle child_scope_handle
        while True:
            child_scope_handle = npi_fsdb_iter_scope_next(child_scope_handle_iter)
            if child_scope_handle == NULL:
                break
            res.append(NpiFsdbScope.init(child_scope_handle))

        return res

    def signal_list(self) -> list[str]:
        assert self.scope_handle != NULL

        res = []
        cdef npiFsdbScopeHandle sig_handle_iter = npi_fsdb_iter_sig(self.scope_handle)
        cdef npiFsdbSigHandle sig_handle
        cdef const char* c_str
        while True:
            sig_handle = npi_fsdb_iter_sig_next(sig_handle_iter)
            if sig_handle == NULL:
                break
            c_str = npi_fsdb_sig_property_str(npiFsdbSigName, sig_handle)
            name = c_str.decode('ascii')
            res.append(name)

        return res

    def name(self) -> str:
        assert self.scope_handle != NULL
        return npi_fsdb_scope_property_str(npiFsdbScopeName, self.scope_handle).decode('ascii')

    def type(self) -> str:
        assert self.scope_handle != NULL
        return npi_fsdb_scope_property_str(npiFsdbScopeType, self.scope_handle).decode('ascii')

    def def_name(self) -> str:
        assert self.scope_handle != NULL
        return npi_fsdb_scope_property_str(npiFsdbScopeDefName, self.scope_handle).decode('ascii')

@cython.boundscheck(False)  # 关闭边界检查以提升性能
@cython.wraparound(False)   # 关闭负索引检查以提升性能
cdef inline cstr_to_ull(char* str, unsigned int xz_value, int max_bit_num = 2147483647):
    cdef unsigned long long value = 0
    cdef int i = 0
    cdef char c
    cdef char zero = b'0'
    while(i < max_bit_num):
        c = str[i]
        if c == b'\0':
            break
        if c == b'0' or c == b'1':
            value = (value << 1) + (c - zero)
        elif c == b'x' or c == b'z' or c == b'X' or c == b'Z':
            value = (value << 1) + xz_value
        elif c == b'{' or c == b'}' or c == b',': # for multi-dimension signal
            pass
        else:
            raise ValueError(f"unknown char '{chr(c)}' in fsdb signal value \"{str.decode('ascii')}\"")
        i = i + 1
    return value

@cython.boundscheck(False)  # 关闭边界检查以提升性能
@cython.wraparound(False)   # 关闭负索引检查以提升性能
cdef inline cstr_to_bit(char* str, unsigned int xz_value):
    cdef unsigned int value = 0
    cdef char c = str[0]
    cdef char zero = b'0'
    if c == b'0' or c == b'1':
        value = (c - zero)
    else:
        value = xz_value
    return value

cdef inline fsdb_read_value_change(vct_handle: npiFsdbVctHandle, begin_time, end_time, width:int, xz_value:int):

    cdef npiFsdbTime cur_time
    cdef npiFsdbValue cur_value
    cdef int stat

    cdef vector[unsigned long long] time_array

    cdef int value_array_num = math.ceil(width/64)
    assert width <= 64*64
    cdef vector[unsigned long long] value_array[64]

    cdef int first = 1
    cdef int i,j
    cdef unsigned long long cur_int_value
    while True:
        if first:
            if begin_time is None:
                stat = npi_fsdb_goto_first(vct_handle)
            else:
                stat = npi_fsdb_goto_time(vct_handle,  <npiFsdbTime> begin_time)
        else:
            stat = npi_fsdb_goto_next(vct_handle)
        if stat == 0:
            break

        cur_value.format = npiFsdbBinStrVal
        npi_fsdb_vct_time(vct_handle, &cur_time)
        npi_fsdb_vct_value(vct_handle, &cur_value)
        if cur_time > end_time:
            break

        time_array.push_back(cur_time)
        if width == 1: # opt for clk
            value_array[0].push_back(cstr_to_bit(<char*>cur_value.value.str, xz_value))
        else:
            cur_lllint_value = 0
            for i in range(value_array_num):
                value_array[i].push_back(cstr_to_ull(<char*>cur_value.value.str + 64*i, xz_value, 64))

    result = np.zeros((value_array[0].size(), 2), dtype=np.uint64 if width <= 64 else np.object_)  # 创建一个 (n, 2) 的二维数组，dtype 为 int32
    for j in range(value_array[0].size()):
        result[j][0] = time_array[j]
        if width <= 64:
            result[j][1] = value_array[0][j]
        else:
            for i in range(value_array_num):
                result[j][1] = (result[j][1] << 64) + value_array[i][j]

    return result

cdef get_signal_handle_width( npiFsdbSigHandle signal_handle):
    cdef int has_member
    cdef int width
    cdef npiFsdbSigIter sub_signal_iter
    cdef npiFsdbSigHandle sub_signal
    assert(npi_fsdb_sig_property(npiFsdbSigHasMember, signal_handle, &has_member))
    if(has_member == 0):
        assert(npi_fsdb_sig_property(npiFsdbSigRangeSize, signal_handle, &width))
        return width
    else:
        sub_signal_iter = npi_fsdb_iter_member(signal_handle)
        assert(sub_signal_iter != NULL)
        width = 0
        while(sub_signal := npi_fsdb_iter_sig_next(sub_signal_iter)):
            width += get_signal_handle_width(sub_signal)
        return width

cdef class NpiFsdbReader:
    cdef npiFsdbFileHandle fsdb_handle
    cdef str file

    def __init__(self, str file):
        #cdef int argc = len(sys.argv)
        #cdef char** argv = <char**>malloc((argc + 1) * sizeof(char*))

        #for i in range(argc):
        #    argv[i] = <char*>malloc(strlen(sys.argv[i].encode('ascii')))
        #    strcpy(argv[i], sys.argv[i].encode('ascii'))
        #argv[argc] = NULL  # Null-terminate the array

        #npi_init(argc, argv)
        #self.file = file
        #free(argv)

        #cdef npiFsdbFileHandle fsdb_handle
        file_str = file.encode('utf-8')
        cdef char* file_s = file_str
        self.fsdb_handle = npi_fsdb_open(file_s)



    def get_signal_width(
        self,
        str signal
    ) -> int:

        cdef npiFsdbSigHandle signal_handle = npi_fsdb_sig_by_name(self.fsdb_handle,  signal.encode('ascii'), NULL)
        assert signal_handle != NULL, f"can't find signal: {signal}"
        return get_signal_handle_width(signal_handle)

    @cython.boundscheck(False)  # 关闭边界检查以提升性能
    @cython.wraparound(False)   # 关闭负索引检查以提升性能
    def load_value_change(
        self,
        str signal,
        unsigned long long begin_time,
        unsigned long long end_time,
        int xz_value
    ) -> np.ndarray:

        cdef npiFsdbSigHandle signal_handle = npi_fsdb_sig_by_name(self.fsdb_handle,  signal.encode('ascii'), NULL)
        assert signal_handle != NULL, f"can't find signal: {signal}"
        cdef int width = get_signal_handle_width(signal_handle)

        cdef npiFsdbVctHandle signal_vct_handle = npi_fsdb_create_vct(signal_handle)
        cdef npiFsdbTime cur_time
        cdef npiFsdbValue cur_value
        cdef int stat

        cdef vector[unsigned long long] time_array

        cdef int value_array_num = (width+63)//64
        assert width <= 64*64
        cdef vector[unsigned long long] value_array[64]

        cdef int first = 1
        cdef int i,j
        while True:
            if first:
                if begin_time is None:
                    stat = npi_fsdb_goto_first(signal_vct_handle)
                else:
                    stat = npi_fsdb_goto_time(signal_vct_handle,  <npiFsdbTime> begin_time)
            else:
                stat = npi_fsdb_goto_next(signal_vct_handle)
            if stat == 0:
                break

            cur_value.format = npiFsdbBinStrVal
            npi_fsdb_vct_time(signal_vct_handle, &cur_time)
            npi_fsdb_vct_value(signal_vct_handle, &cur_value)

            if cur_time > end_time:
                break
            time_array.push_back(cur_time)

            if width == 1: # opt for clk
                value_array[0].push_back(cstr_to_bit(<char*>cur_value.value.str, xz_value))
            else:
                for i in range(value_array_num):
                    value_array[i].push_back(cstr_to_ull(<char*>cur_value.value.str + 64*i, xz_value, 64))
            first = False

        cdef np.ndarray[np.uint64_t, ndim=2] result_uint64 = np.zeros((value_array[0].size(), 2), dtype=np.uint64)
        cdef result_object = np.zeros((value_array[0].size(), 2), dtype=np.object_)
        cdef unsigned int shift
        if width <= 64:
            value_np_array = np.PyArray_SimpleNewFromData(1, [value_array[0].size()], np.NPY_UINT64, value_array[0].data())
            time_np_array =  np.PyArray_SimpleNewFromData(1, [time_array.size()], np.NPY_UINT64, time_array.data())

            result_uint64[:,1] = value_np_array
            result_uint64[:,0] = time_np_array
            return result_uint64

        else:
            time_np_array =  np.PyArray_SimpleNewFromData(1, [time_array.size()], np.NPY_UINT64, time_array.data())
            result_object[:,0] = time_np_array
            for i in range(value_array_num):
                shift = min(64, width - i * 64)
                value_np_array = np.PyArray_SimpleNewFromData(1, [value_array[i].size()], np.NPY_UINT64, value_array[i].data())
                result_object[:,1] = (result_object[:,1] << shift) + value_np_array
            return result_object

        #npi_fsdb_release_vct(signal_vct_handle)
        #npi_fsdb_unload_vc(self.fsdb_handle)
        #return result

    def top_scope_list(self) -> list[NpiFsdbScope]:
        cdef npiFsdbScopeIter top_scope_iter
        cdef npiFsdbScopeHandle top_scope_handle

        res = []
        top_scope_iter = npi_fsdb_iter_top_scope(self.fsdb_handle)
        while True:
            top_scope_handle = npi_fsdb_iter_scope_next(top_scope_iter)
            if top_scope_handle == NULL:
                break
            res.append(NpiFsdbScope.init(top_scope_handle))
        return res

    def min_time(self) -> int:
        cdef npiFsdbTime time
        npi_fsdb_min_time(self.fsdb_handle, &time)
        return <int>time

    def max_time(self) -> int:
        cdef npiFsdbTime time
        npi_fsdb_max_time(self.fsdb_handle, &time)
        return <int>time

    def close(self):
        npi_fsdb_close(self.fsdb_handle)
