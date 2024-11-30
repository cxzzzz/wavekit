from libc.stdint cimport uint64_t
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)  # 关闭边界检查以提升性能
@cython.wraparound(False)   # 关闭负索引检查以提升性能
def str_value_change_to_int_value_change_uint64(list[tuple[int, str]] int_str_list, int xz_value):
    cdef int i, j, n, length
    cdef bytes s
    cdef np.ndarray[np.uint64_t, ndim=2] result
    cdef int int_part
    cdef char c
    cdef char z_char = b'z'[0]
    cdef char x_char = b'x'[0]
    cdef char Z_char = b'Z'[0]
    cdef char X_char = b'X'[0]
    cdef char zero_char = b'0'[0]
    cdef uint64_t new_value

    n = len(int_str_list)
    result = np.zeros((n, 2), dtype=np.uint64)  # 创建一个 (n, 2) 的二维数组，dtype 为 int32

    for i in range(n):
        int_part, s = int_str_list[i][0] , int_str_list[i][1].encode('ascii') # 将字符串编码为字节
        length = len(s)
        new_value = 0

        for j in range(length):
            c = s[j]
            if c == z_char or c == x_char or c == Z_char or c == X_char:
                new_value = (new_value << 1) + xz_value
            else:
                new_value = (new_value << 1) + c - zero_char

        # 填充结果数dw
        result[i, 0] = int_part
        result[i, 1] = new_value

    return result

@cython.boundscheck(False)  # 关闭边界检查以提升性能
@cython.wraparound(False)   # 关闭负索引检查以提升性能
def value_change_to_value_array_uint64(
        np.ndarray[np.uint64_t, ndim=2] value_change,
        np.ndarray[np.uint64_t, ndim=2] clock_changes,
        bint sample_on_posedge
    )-> tuple[np.ndarray, np.ndarray, np.ndarray]:

        cdef np.ndarray[np.uint64_t, ndim=1] value_time = value_change[:,0]
        cdef np.ndarray[np.uint64_t, ndim=1] value = value_change[:,1]
        cdef np.ndarray[np.uint64_t, ndim=1] clock_time = clock_changes[:,0]
        cdef np.ndarray[np.uint64_t, ndim=1] clock = clock_changes[:,1]

        cdef np.ndarray[np.uint64_t, ndim=1] value_res = np.zeros([clock.size], dtype=np.uint64)
        cdef np.ndarray[np.uint64_t, ndim=1] clock_res = np.zeros([clock.size], dtype=np.uint64)
        cdef np.ndarray[np.uint64_t, ndim=1] time_res = np.zeros([clock.size], dtype=np.uint64)

        cdef int sample_clock_value = 1 if sample_on_posedge else 0
        cdef unsigned long vidx = 0, cidx = 0 , ccnt = 0
        cdef unsigned long crange = clock.size, vrange = value.size
        cdef unsigned long cvalue, ctime
        for cidx in range(crange):
            cvalue = clock[cidx]
            if cvalue == sample_clock_value:
                ctime = clock_time[cidx]
                while vidx + 1 < vrange and value_time[vidx + 1] <= ctime:
                    vidx += 1
                value_res[ccnt]=value[vidx]
                clock_res[ccnt]=ccnt
                time_res[ccnt]=ctime
                ccnt += 1

        return (
            value_res[:ccnt], clock_res[:ccnt], time_res[:ccnt]
        )

@cython.boundscheck(False)  # 关闭边界检查以提升性能
@cython.wraparound(False)   # 关闭负索引检查以提升性能
def value_change_to_value_array_object(
        np.ndarray value_change,
        np.ndarray[np.uint64_t, ndim=2] clock_changes,
        bint sample_on_posedge
    )-> tuple[np.ndarray, np.ndarray, np.ndarray]:

        cdef np.ndarray[object, ndim=1] value = value_change[:,1]
        cdef np.ndarray[np.uint64_t, ndim=1] value_time = value_change[:,0].astype(np.uint64)
        cdef np.ndarray[np.uint64_t, ndim=1] clock_time = clock_changes[:,0]
        cdef np.ndarray[np.uint64_t, ndim=1] clock = clock_changes[:,1]

        value_res = np.zeros([clock.size], dtype=np.object_)
        cdef np.ndarray[np.uint64_t, ndim=1] clock_res = np.zeros([clock.size], dtype=np.uint64)
        cdef np.ndarray[np.uint64_t, ndim=1] time_res = np.zeros([clock.size], dtype=np.uint64)

        cdef int sample_clock_value = 1 if sample_on_posedge else 0
        cdef unsigned long vidx = 0, cidx = 0 , ccnt = 0
        cdef unsigned long crange = clock.size, vrange = value.size
        cdef unsigned long cvalue, ctime
        for cidx in range(crange):
            cvalue = clock[cidx]
            if cvalue == sample_clock_value:
                ctime = clock_time[cidx]
                while vidx + 1 < vrange and value_time[vidx + 1] <= ctime:
                    vidx += 1
                value_res[ccnt]=value[vidx]
                clock_res[ccnt]=ccnt
                time_res[ccnt]=ctime
                ccnt += 1

        return (
            value_res[:ccnt], clock_res[:ccnt], time_res[:ccnt]
        )

def value_change_to_value_array(
    np.ndarray value_change,
    np.ndarray clock_changes,
    bint sample_on_posedge
):
    if value_change.dtype == np.object_:
        return value_change_to_value_array_object(value_change, clock_changes, sample_on_posedge)
    else:
        return value_change_to_value_array_uint64(value_change, clock_changes, sample_on_posedge)
