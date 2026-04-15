# cython: language_level=3
import ctypes.util
import os
import numpy as np
cimport numpy as np
import cython
from libcpp.vector cimport vector
from .npi_fsdb cimport *

cdef extern from "dlfcn.h":
    enum:
        RTLD_NOW
        RTLD_LOCAL

    void* dlopen(const char* filename, int flags)
    void* dlsym(void* handle, const char* symbol)
    int dlclose(void* handle)
    const char* dlerror()

ctypedef npiFsdbFileHandle (*npi_fsdb_open_fn)(const NPI_BYTE8* name)
ctypedef NPI_INT32 (*npi_fsdb_close_fn)(npiFsdbFileHandle file)
ctypedef NPI_INT32 (*npi_fsdb_min_time_fn)(npiFsdbFileHandle file, npiFsdbTime* time)
ctypedef NPI_INT32 (*npi_fsdb_max_time_fn)(npiFsdbFileHandle file, npiFsdbTime* time)
ctypedef npiFsdbSigHandle (*npi_fsdb_sig_by_name_fn)(
    npiFsdbFileHandle file,
    const NPI_BYTE8* name,
    npiFsdbScopeHandle scope,
)
ctypedef npiFsdbVctHandle (*npi_fsdb_create_vct_fn)(npiFsdbSigHandle sig)
ctypedef NPI_INT32 (*npi_fsdb_goto_time_fn)(npiFsdbVctHandle vct, npiFsdbTime time)
ctypedef NPI_INT32 (*npi_fsdb_goto_first_fn)(npiFsdbVctHandle vct)
ctypedef NPI_INT32 (*npi_fsdb_goto_next_fn)(npiFsdbVctHandle vct)
ctypedef NPI_INT32 (*npi_fsdb_vct_time_fn)(npiFsdbVctHandle vct, npiFsdbTime* time)
ctypedef NPI_INT32 (*npi_fsdb_vct_value_fn)(npiFsdbVctHandle vct, npiFsdbValue* value)
ctypedef NPI_INT32 (*npi_fsdb_sig_property_fn)(
    npiFsdbSigPropertyType type,
    npiFsdbSigHandle sig,
    NPI_INT32* prop,
)
ctypedef const NPI_BYTE8* (*npi_fsdb_sig_property_str_fn)(
    npiFsdbSigPropertyType type,
    npiFsdbSigHandle sig,
)
ctypedef const NPI_BYTE8* (*npi_fsdb_scope_property_str_fn)(
    npiFsdbScopePropertyType type,
    npiFsdbScopeHandle scope,
)
ctypedef npiFsdbScopeIter (*npi_fsdb_iter_top_scope_fn)(npiFsdbFileHandle file)
ctypedef npiFsdbScopeIter (*npi_fsdb_iter_child_scope_fn)(npiFsdbScopeHandle scope)
ctypedef npiFsdbScopeHandle (*npi_fsdb_iter_scope_next_fn)(npiFsdbScopeIter iter)
ctypedef npiFsdbSigHandle (*npi_fsdb_iter_sig_next_fn)(npiFsdbSigIter iter)
ctypedef npiFsdbSigIter (*npi_fsdb_iter_sig_fn)(npiFsdbScopeHandle scope)
ctypedef npiFsdbSigIter (*npi_fsdb_iter_member_fn)(npiFsdbSigHandle sig)
ctypedef NPI_INT32 (*npi_fsdb_iter_scope_stop_fn)(npiFsdbScopeIter iter)
ctypedef NPI_INT32 (*npi_fsdb_iter_sig_stop_fn)(npiFsdbSigIter iter)
ctypedef NPI_INT32 (*npi_fsdb_release_vct_fn)(npiFsdbVctHandle vct)

cdef void* _npi_lib_handle = NULL
cdef object _npi_lib_path = None
cdef npi_fsdb_open_fn _npi_fsdb_open = NULL
cdef npi_fsdb_close_fn _npi_fsdb_close = NULL
cdef npi_fsdb_min_time_fn _npi_fsdb_min_time = NULL
cdef npi_fsdb_max_time_fn _npi_fsdb_max_time = NULL
cdef npi_fsdb_sig_by_name_fn _npi_fsdb_sig_by_name = NULL
cdef npi_fsdb_create_vct_fn _npi_fsdb_create_vct = NULL
cdef npi_fsdb_goto_time_fn _npi_fsdb_goto_time = NULL
cdef npi_fsdb_goto_first_fn _npi_fsdb_goto_first = NULL
cdef npi_fsdb_goto_next_fn _npi_fsdb_goto_next = NULL
cdef npi_fsdb_vct_time_fn _npi_fsdb_vct_time = NULL
cdef npi_fsdb_vct_value_fn _npi_fsdb_vct_value = NULL
cdef npi_fsdb_sig_property_fn _npi_fsdb_sig_property = NULL
cdef npi_fsdb_sig_property_str_fn _npi_fsdb_sig_property_str = NULL
cdef npi_fsdb_scope_property_str_fn _npi_fsdb_scope_property_str = NULL
cdef npi_fsdb_iter_top_scope_fn _npi_fsdb_iter_top_scope = NULL
cdef npi_fsdb_iter_child_scope_fn _npi_fsdb_iter_child_scope = NULL
cdef npi_fsdb_iter_scope_next_fn _npi_fsdb_iter_scope_next = NULL
cdef npi_fsdb_iter_sig_next_fn _npi_fsdb_iter_sig_next = NULL
cdef npi_fsdb_iter_sig_fn _npi_fsdb_iter_sig = NULL
cdef npi_fsdb_iter_member_fn _npi_fsdb_iter_member = NULL
cdef npi_fsdb_iter_scope_stop_fn _npi_fsdb_iter_scope_stop = NULL
cdef npi_fsdb_iter_sig_stop_fn _npi_fsdb_iter_sig_stop = NULL
cdef npi_fsdb_release_vct_fn _npi_fsdb_release_vct = NULL

# Verdi's libNPI.so exports these APIs as C++ symbols on some releases, so the
# runtime binder must fall back to the mangled names when the plain C names are
# not present.
cdef dict _NPI_CPP_SYMBOL_ALIASES = {
    b'npi_fsdb_open': (b'_Z13npi_fsdb_openPKc',),
    b'npi_fsdb_close': (b'_Z14npi_fsdb_closePv',),
    b'npi_fsdb_min_time': (b'_Z17npi_fsdb_min_timePvPy',),
    b'npi_fsdb_max_time': (b'_Z17npi_fsdb_max_timePvPy',),
    b'npi_fsdb_sig_by_name': (b'_Z20npi_fsdb_sig_by_namePvPKcS_',),
    b'npi_fsdb_create_vct': (b'_Z19npi_fsdb_create_vctPv',),
    b'npi_fsdb_goto_time': (b'_Z18npi_fsdb_goto_timePvy',),
    b'npi_fsdb_goto_first': (b'_Z19npi_fsdb_goto_firstPv',),
    b'npi_fsdb_goto_next': (b'_Z18npi_fsdb_goto_nextPv',),
    b'npi_fsdb_vct_time': (b'_Z17npi_fsdb_vct_timePvPy',),
    b'npi_fsdb_vct_value': (b'_Z18npi_fsdb_vct_valuePvP12npiFsdbValue',),
    b'npi_fsdb_sig_property': (b'_Z21npi_fsdb_sig_property22npiFsdbSigPropertyTypePvPi',),
    b'npi_fsdb_sig_property_str': (b'_Z25npi_fsdb_sig_property_str22npiFsdbSigPropertyTypePv',),
    b'npi_fsdb_scope_property_str': (b'_Z27npi_fsdb_scope_property_str24npiFsdbScopePropertyTypePv',),
    b'npi_fsdb_iter_top_scope': (b'_Z23npi_fsdb_iter_top_scopePv',),
    b'npi_fsdb_iter_child_scope': (b'_Z25npi_fsdb_iter_child_scopePv',),
    b'npi_fsdb_iter_scope_next': (b'_Z24npi_fsdb_iter_scope_nextPv',),
    b'npi_fsdb_iter_sig_next': (b'_Z22npi_fsdb_iter_sig_nextPv',),
    b'npi_fsdb_iter_sig': (b'_Z17npi_fsdb_iter_sigPv',),
    b'npi_fsdb_iter_member': (b'_Z20npi_fsdb_iter_memberPv',),
    b'npi_fsdb_iter_scope_stop': (b'_Z24npi_fsdb_iter_scope_stopPv',),
    b'npi_fsdb_iter_sig_stop': (b'_Z22npi_fsdb_iter_sig_stopPv',),
    b'npi_fsdb_release_vct': (b'_Z20npi_fsdb_release_vctPv',),
}


cdef str _decode_cstr(const char* value):
    if value == NULL:
        return ''
    return (<bytes>value).decode('utf-8', 'replace')


cdef str _last_dlerror():
    return _decode_cstr(dlerror())


cdef tuple _symbol_candidates(object symbol_name):
    cdef bytes symbol_bytes
    cdef tuple aliases

    symbol_bytes = symbol_name if isinstance(symbol_name, bytes) else str(symbol_name).encode('ascii')
    aliases = _NPI_CPP_SYMBOL_ALIASES.get(symbol_bytes, ())
    return (symbol_bytes,) + aliases


cdef void _clear_npi_symbols():
    global _npi_fsdb_open, _npi_fsdb_close, _npi_fsdb_min_time, _npi_fsdb_max_time
    global _npi_fsdb_sig_by_name, _npi_fsdb_create_vct, _npi_fsdb_goto_time
    global _npi_fsdb_goto_first, _npi_fsdb_goto_next, _npi_fsdb_vct_time
    global _npi_fsdb_vct_value, _npi_fsdb_sig_property, _npi_fsdb_sig_property_str
    global _npi_fsdb_scope_property_str, _npi_fsdb_iter_top_scope
    global _npi_fsdb_iter_child_scope, _npi_fsdb_iter_scope_next, _npi_fsdb_iter_sig_next
    global _npi_fsdb_iter_sig, _npi_fsdb_iter_member, _npi_fsdb_iter_scope_stop
    global _npi_fsdb_iter_sig_stop, _npi_fsdb_release_vct
    _npi_fsdb_open = NULL
    _npi_fsdb_close = NULL
    _npi_fsdb_min_time = NULL
    _npi_fsdb_max_time = NULL
    _npi_fsdb_sig_by_name = NULL
    _npi_fsdb_create_vct = NULL
    _npi_fsdb_goto_time = NULL
    _npi_fsdb_goto_first = NULL
    _npi_fsdb_goto_next = NULL
    _npi_fsdb_vct_time = NULL
    _npi_fsdb_vct_value = NULL
    _npi_fsdb_sig_property = NULL
    _npi_fsdb_sig_property_str = NULL
    _npi_fsdb_scope_property_str = NULL
    _npi_fsdb_iter_top_scope = NULL
    _npi_fsdb_iter_child_scope = NULL
    _npi_fsdb_iter_scope_next = NULL
    _npi_fsdb_iter_sig_next = NULL
    _npi_fsdb_iter_sig = NULL
    _npi_fsdb_iter_member = NULL
    _npi_fsdb_iter_scope_stop = NULL
    _npi_fsdb_iter_sig_stop = NULL
    _npi_fsdb_release_vct = NULL


cdef void* _require_symbol(void* handle, object symbol_name) except NULL:
    cdef void* symbol
    cdef tuple symbol_candidates = _symbol_candidates(symbol_name)
    cdef bytes candidate
    cdef const char* candidate_name
    cdef list tried_names = []

    for candidate in symbol_candidates:
        candidate_name = candidate
        dlerror()
        symbol = dlsym(handle, candidate_name)
        if symbol != NULL:
            return symbol
        tried_names.append(repr(candidate.decode('ascii')))

    raise OSError(
        f"Failed to resolve symbol {symbol_candidates[0].decode('ascii')!r} from "
        f"{_npi_lib_path or 'libNPI.so'}; tried {', '.join(tried_names)}: "
        f"{_last_dlerror() or 'symbol not found'}"
    )


cdef void _bind_npi_symbols(void* handle) except *:
    global _npi_fsdb_open, _npi_fsdb_close, _npi_fsdb_min_time, _npi_fsdb_max_time
    global _npi_fsdb_sig_by_name, _npi_fsdb_create_vct, _npi_fsdb_goto_time
    global _npi_fsdb_goto_first, _npi_fsdb_goto_next, _npi_fsdb_vct_time
    global _npi_fsdb_vct_value, _npi_fsdb_sig_property, _npi_fsdb_sig_property_str
    global _npi_fsdb_scope_property_str, _npi_fsdb_iter_top_scope
    global _npi_fsdb_iter_child_scope, _npi_fsdb_iter_scope_next, _npi_fsdb_iter_sig_next
    global _npi_fsdb_iter_sig, _npi_fsdb_iter_member, _npi_fsdb_iter_scope_stop
    global _npi_fsdb_iter_sig_stop, _npi_fsdb_release_vct
    _npi_fsdb_open = <npi_fsdb_open_fn>_require_symbol(handle, b'npi_fsdb_open')
    _npi_fsdb_close = <npi_fsdb_close_fn>_require_symbol(handle, b'npi_fsdb_close')
    _npi_fsdb_min_time = <npi_fsdb_min_time_fn>_require_symbol(handle, b'npi_fsdb_min_time')
    _npi_fsdb_max_time = <npi_fsdb_max_time_fn>_require_symbol(handle, b'npi_fsdb_max_time')
    _npi_fsdb_sig_by_name = <npi_fsdb_sig_by_name_fn>_require_symbol(
        handle, b'npi_fsdb_sig_by_name'
    )
    _npi_fsdb_create_vct = <npi_fsdb_create_vct_fn>_require_symbol(handle, b'npi_fsdb_create_vct')
    _npi_fsdb_goto_time = <npi_fsdb_goto_time_fn>_require_symbol(handle, b'npi_fsdb_goto_time')
    _npi_fsdb_goto_first = <npi_fsdb_goto_first_fn>_require_symbol(handle, b'npi_fsdb_goto_first')
    _npi_fsdb_goto_next = <npi_fsdb_goto_next_fn>_require_symbol(handle, b'npi_fsdb_goto_next')
    _npi_fsdb_vct_time = <npi_fsdb_vct_time_fn>_require_symbol(handle, b'npi_fsdb_vct_time')
    _npi_fsdb_vct_value = <npi_fsdb_vct_value_fn>_require_symbol(handle, b'npi_fsdb_vct_value')
    _npi_fsdb_sig_property = <npi_fsdb_sig_property_fn>_require_symbol(
        handle, b'npi_fsdb_sig_property'
    )
    _npi_fsdb_sig_property_str = <npi_fsdb_sig_property_str_fn>_require_symbol(
        handle, b'npi_fsdb_sig_property_str'
    )
    _npi_fsdb_scope_property_str = <npi_fsdb_scope_property_str_fn>_require_symbol(
        handle, b'npi_fsdb_scope_property_str'
    )
    _npi_fsdb_iter_top_scope = <npi_fsdb_iter_top_scope_fn>_require_symbol(
        handle, b'npi_fsdb_iter_top_scope'
    )
    _npi_fsdb_iter_child_scope = <npi_fsdb_iter_child_scope_fn>_require_symbol(
        handle, b'npi_fsdb_iter_child_scope'
    )
    _npi_fsdb_iter_scope_next = <npi_fsdb_iter_scope_next_fn>_require_symbol(
        handle, b'npi_fsdb_iter_scope_next'
    )
    _npi_fsdb_iter_sig_next = <npi_fsdb_iter_sig_next_fn>_require_symbol(
        handle, b'npi_fsdb_iter_sig_next'
    )
    _npi_fsdb_iter_sig = <npi_fsdb_iter_sig_fn>_require_symbol(handle, b'npi_fsdb_iter_sig')
    _npi_fsdb_iter_member = <npi_fsdb_iter_member_fn>_require_symbol(handle, b'npi_fsdb_iter_member')
    _npi_fsdb_iter_scope_stop = <npi_fsdb_iter_scope_stop_fn>_require_symbol(
        handle, b'npi_fsdb_iter_scope_stop'
    )
    _npi_fsdb_iter_sig_stop = <npi_fsdb_iter_sig_stop_fn>_require_symbol(
        handle, b'npi_fsdb_iter_sig_stop'
    )
    _npi_fsdb_release_vct = <npi_fsdb_release_vct_fn>_require_symbol(
        handle, b'npi_fsdb_release_vct'
    )


cdef list _npi_library_candidates(object preferred_path):
    cdef list candidates = []
    cdef set seen = set()
    cdef object candidate
    verdi_home = os.environ.get('VERDI_HOME')

    for candidate in [
        preferred_path,
        os.environ.get('WAVEKIT_NPI_LIB'),
        os.path.join(verdi_home, 'share', 'NPI', 'lib', 'LINUX64', 'libNPI.so')
        if verdi_home
        else None,
        os.path.join(verdi_home, 'share', 'NPI', 'lib', 'linux64', 'libNPI.so')
        if verdi_home
        else None,
        os.path.join(verdi_home, 'share', 'NPI', 'lib', 'libNPI.so') if verdi_home else None,
        ctypes.util.find_library('NPI'),
        'libNPI.so',
    ]:
        if candidate and candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)
    return candidates


cdef void _ensure_npi_loaded(object preferred_path=None) except *:
    global _npi_lib_handle, _npi_lib_path

    cdef void* handle
    cdef list errors = []
    cdef bytes path_bytes
    cdef object candidate

    if _npi_lib_handle != NULL:
        return

    for candidate in _npi_library_candidates(preferred_path):
        path_bytes = str(candidate).encode('utf-8')
        handle = dlopen(path_bytes, RTLD_NOW | RTLD_LOCAL)
        if handle == NULL:
            errors.append(f'{candidate}: {_last_dlerror() or "dlopen failed"}')
            continue
        try:
            _npi_lib_path = str(candidate)
            _bind_npi_symbols(handle)
        except Exception as exc:
            _clear_npi_symbols()
            dlclose(handle)
            errors.append(f'{candidate}: {exc}')
            _npi_lib_path = None
            continue
        _npi_lib_handle = handle
        return

    raise OSError(
        'Failed to load Verdi FSDB runtime library (libNPI.so). Configure via:\n'
        '  - WAVEKIT_NPI_LIB — direct path to libNPI.so\n'
        '  - VERDI_HOME — Verdi installation directory\n'
        '  - LD_LIBRARY_PATH — system library search path\n'
        + '\n'.join(errors)
    )


cpdef bint fsdb_runtime_available(object preferred_path=None):
    try:
        _ensure_npi_loaded(preferred_path)
        return True
    except Exception:
        return False


cpdef object fsdb_runtime_library_path():
    return _npi_lib_path


cdef inline npiFsdbFileHandle npi_fsdb_open(const NPI_BYTE8* name):
    return _npi_fsdb_open(name)


cdef inline NPI_INT32 npi_fsdb_close(npiFsdbFileHandle file):
    return _npi_fsdb_close(file)


cdef inline NPI_INT32 npi_fsdb_min_time(npiFsdbFileHandle file, npiFsdbTime* time):
    return _npi_fsdb_min_time(file, time)


cdef inline NPI_INT32 npi_fsdb_max_time(npiFsdbFileHandle file, npiFsdbTime* time):
    return _npi_fsdb_max_time(file, time)


cdef inline npiFsdbSigHandle npi_fsdb_sig_by_name(
    npiFsdbFileHandle file,
    const NPI_BYTE8* name,
    npiFsdbScopeHandle scope,
):
    return _npi_fsdb_sig_by_name(file, name, scope)


cdef inline npiFsdbVctHandle npi_fsdb_create_vct(npiFsdbSigHandle sig):
    return _npi_fsdb_create_vct(sig)


cdef inline NPI_INT32 npi_fsdb_goto_time(npiFsdbVctHandle vct, npiFsdbTime time):
    return _npi_fsdb_goto_time(vct, time)


cdef inline NPI_INT32 npi_fsdb_goto_first(npiFsdbVctHandle vct):
    return _npi_fsdb_goto_first(vct)


cdef inline NPI_INT32 npi_fsdb_goto_next(npiFsdbVctHandle vct):
    return _npi_fsdb_goto_next(vct)


cdef inline NPI_INT32 npi_fsdb_vct_time(npiFsdbVctHandle vct, npiFsdbTime* time):
    return _npi_fsdb_vct_time(vct, time)


cdef inline NPI_INT32 npi_fsdb_vct_value(npiFsdbVctHandle vct, npiFsdbValue* value):
    return _npi_fsdb_vct_value(vct, value)


cdef inline NPI_INT32 npi_fsdb_sig_property(
    npiFsdbSigPropertyType type,
    npiFsdbSigHandle sig,
    NPI_INT32* prop,
):
    return _npi_fsdb_sig_property(type, sig, prop)


cdef inline const NPI_BYTE8* npi_fsdb_sig_property_str(
    npiFsdbSigPropertyType type,
    npiFsdbSigHandle sig,
):
    return _npi_fsdb_sig_property_str(type, sig)


cdef inline const NPI_BYTE8* npi_fsdb_scope_property_str(
    npiFsdbScopePropertyType type,
    npiFsdbScopeHandle scope,
):
    return _npi_fsdb_scope_property_str(type, scope)


cdef inline npiFsdbScopeIter npi_fsdb_iter_top_scope(npiFsdbFileHandle file):
    return _npi_fsdb_iter_top_scope(file)


cdef inline npiFsdbScopeIter npi_fsdb_iter_child_scope(npiFsdbScopeHandle scope):
    return _npi_fsdb_iter_child_scope(scope)


cdef inline npiFsdbScopeHandle npi_fsdb_iter_scope_next(npiFsdbScopeIter iter):
    return _npi_fsdb_iter_scope_next(iter)


cdef inline npiFsdbSigHandle npi_fsdb_iter_sig_next(npiFsdbSigIter iter):
    return _npi_fsdb_iter_sig_next(iter)


cdef inline npiFsdbSigIter npi_fsdb_iter_sig(npiFsdbScopeHandle scope):
    return _npi_fsdb_iter_sig(scope)


cdef inline npiFsdbSigIter npi_fsdb_iter_member(npiFsdbSigHandle sig):
    return _npi_fsdb_iter_member(sig)


cdef inline NPI_INT32 npi_fsdb_iter_scope_stop(npiFsdbScopeIter iter):
    return _npi_fsdb_iter_scope_stop(iter)


cdef inline NPI_INT32 npi_fsdb_iter_sig_stop(npiFsdbSigIter iter):
    return _npi_fsdb_iter_sig_stop(iter)


cdef inline NPI_INT32 npi_fsdb_release_vct(npiFsdbVctHandle vct):
    return _npi_fsdb_release_vct(vct)

NPI_FSDB_CT_ARRAY        = <int>npiFsdbSigCtArray
NPI_FSDB_CT_STRUCT       = <int>npiFsdbSigCtStruct
NPI_FSDB_CT_UNION        = <int>npiFsdbSigCtUnion
NPI_FSDB_CT_TAGGED_UNION = <int>npiFsdbSigCtTaggedUnion
NPI_FSDB_CT_RECORD       = <int>npiFsdbSigCtRecord


cdef class NpiFsdbSignal:
    """Wraps a npiFsdbSigHandle — a single signal (leaf or composite) in the FSDB hierarchy."""
    cdef npiFsdbSigHandle sig_handle

    @staticmethod
    cdef init(npiFsdbSigHandle sig_handle):
        sig = NpiFsdbSignal()
        sig.sig_handle = sig_handle
        return sig

    def __init__(self):
        self.sig_handle = NULL

    def name(self) -> str:
        assert self.sig_handle != NULL
        name = npi_fsdb_sig_property_str(npiFsdbSigName, self.sig_handle).decode('ascii')
        # NPI returns fully-qualified names like "tb.dut.field"; take only the last component.
        # Array elements are named "a[0]", "a[0][1]" etc. — splitting by "." is safe since
        # brackets cannot contain ".".
        name = name.split(".")
        name = name[len(name)-1]
        return name

    def has_member(self) -> bool:
        assert self.sig_handle != NULL
        cdef int has_member
        if npi_fsdb_sig_property(npiFsdbSigHasMember, self.sig_handle, &has_member):
            return has_member != 0
        return False

    def composite_type(self):
        """Return the npiFsdbSigCompositeType_e int value, or None if not composite."""
        assert self.sig_handle != NULL
        cdef int has_member
        cdef int ct
        if not (npi_fsdb_sig_property(npiFsdbSigHasMember, self.sig_handle, &has_member) and has_member):
            return None
        if npi_fsdb_sig_property(npiFsdbSigCompositeType, self.sig_handle, &ct):
            return ct
        return None

    def width(self) -> int:
        """Return the total bit-width of this signal (sum of all leaf members for composites)."""
        assert self.sig_handle != NULL
        cdef int has_member
        cdef int w
        cdef npiFsdbSigIter sub_signal_iter
        cdef npiFsdbSigHandle sub_signal
        assert npi_fsdb_sig_property(npiFsdbSigHasMember, self.sig_handle, &has_member)
        if has_member == 0:
            assert npi_fsdb_sig_property(npiFsdbSigRangeSize, self.sig_handle, &w)
            return w
        else:
            sub_signal_iter = npi_fsdb_iter_member(self.sig_handle)
            assert sub_signal_iter != NULL
            w = 0
            while (sub_signal := npi_fsdb_iter_sig_next(sub_signal_iter)):
                w += NpiFsdbSignal.init(sub_signal).width()
            return w

    def range(self):
        """Return the (high, low) bit-range tuple, or None for non-array composites."""
        assert self.sig_handle != NULL
        cdef int has_member
        cdef int ct
        cdef int left_range, right_range
        assert npi_fsdb_sig_property(npiFsdbSigHasMember, self.sig_handle, &has_member)
        assert npi_fsdb_sig_property(npiFsdbSigLeftRange, self.sig_handle, &left_range)
        assert npi_fsdb_sig_property(npiFsdbSigRightRange, self.sig_handle, &right_range)
        if has_member != 0:
            assert npi_fsdb_sig_property(npiFsdbSigCompositeType, self.sig_handle, &ct)
            if ct == <int>npiFsdbSigCtArray:
                return (left_range, right_range)
            return None
        return None

    def member_list(self) -> list:
        """Return direct member NpiFsdbSignal objects for composite signals."""
        assert self.sig_handle != NULL
        cdef int has_member
        cdef npiFsdbSigIter member_iter
        cdef npiFsdbSigHandle member_handle

        res = []
        if npi_fsdb_sig_property(npiFsdbSigHasMember, self.sig_handle, &has_member) and has_member:
            member_iter = npi_fsdb_iter_member(self.sig_handle)
            while True:
                member_handle = npi_fsdb_iter_sig_next(member_iter)
                if member_handle == NULL:
                    break
                res.append(NpiFsdbSignal.init(member_handle))
            npi_fsdb_iter_sig_stop(member_iter)
        return res


cdef class NpiFsdbScope:
    """Wraps a npiFsdbScopeHandle — a scope node (module/block) in the FSDB hierarchy."""
    cdef npiFsdbScopeHandle scope_handle

    @staticmethod
    cdef init(npiFsdbScopeHandle scope_handle):
        scope = NpiFsdbScope()
        scope.scope_handle = scope_handle
        return scope

    def __init__(self):
        self.scope_handle = NULL

    def child_scope_list(self) -> list:
        """Return direct child NpiFsdbScope nodes (not signals)."""
        assert self.scope_handle != NULL
        res = []
        child_scope_handle_iter = npi_fsdb_iter_child_scope(self.scope_handle)

        cdef npiFsdbScopeHandle child_scope_handle
        while True:
            child_scope_handle = npi_fsdb_iter_scope_next(child_scope_handle_iter)
            if child_scope_handle == NULL:
                break
            res.append(NpiFsdbScope.init(child_scope_handle))
        npi_fsdb_iter_scope_stop(child_scope_handle_iter)
        return res

    def signal_list(self) -> list:
        """Return direct child NpiFsdbSignal objects declared in this scope."""
        assert self.scope_handle != NULL
        res = []
        child_sig_handle_iter = npi_fsdb_iter_sig(self.scope_handle)

        cdef npiFsdbSigHandle child_sig_handle
        while True:
            child_sig_handle = npi_fsdb_iter_sig_next(child_sig_handle_iter)
            if child_sig_handle == NULL:
                break
            res.append(NpiFsdbSignal.init(child_sig_handle))
        npi_fsdb_iter_sig_stop(child_sig_handle_iter)
        return res

    def name(self) -> str:
        assert self.scope_handle != NULL
        return npi_fsdb_scope_property_str(npiFsdbScopeName, self.scope_handle).decode('ascii')

    def type(self) -> str:
        assert self.scope_handle != NULL
        return npi_fsdb_scope_property_str(npiFsdbScopeType, self.scope_handle).decode('ascii')

    def def_name(self) -> str | None:
        """Return the module definition name, or None if this scope is not a module."""
        if self.type() != 'npiFsdbScopeSvModule':
            return None
        return npi_fsdb_scope_property_str(npiFsdbScopeDefName, self.scope_handle).decode('ascii')


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int cstr_to_ull_array(char* str, unsigned int xz_value, unsigned long long* result, int total_bits):
    cdef int bit_count = 0
    cdef int pos = 0
    cdef int chunk_idx = 0
    cdef int bit_in_chunk = 0
    cdef char c
    cdef char zero = b'0'

    for i in range((total_bits + 63) // 64):
        result[i] = 0

    while bit_count < total_bits:
        c = str[pos]
        if c == b'\0':
            break
        if c == b'0' or c == b'1':
            result[chunk_idx] = (result[chunk_idx] << 1) + (c - zero)
            bit_in_chunk += 1
            bit_count += 1
        elif c == b'x' or c == b'z' or c == b'X' or c == b'Z':
            result[chunk_idx] = (result[chunk_idx] << 1) + xz_value
            bit_in_chunk += 1
            bit_count += 1
        elif c == b'{' or c == b'}' or c == b',':
            pass
        else:
            raise ValueError(f"unknown char '{chr(c)}' in fsdb signal value \"{str.decode('ascii')}\"")
        if bit_in_chunk == 64:
            chunk_idx += 1
            bit_in_chunk = 0
        pos += 1

    return bit_count


cdef class NpiFsdbReader:
    cdef npiFsdbFileHandle fsdb_handle
    cdef str file

    def __init__(self, str file):
        _ensure_npi_loaded()

        file_str = file.encode('utf-8')
        cdef char* file_s = file_str
        self.fsdb_handle = npi_fsdb_open(file_s)
        if(self.fsdb_handle == NULL):
            raise OSError(
                f"Failed to open fsdb file {file!r} using {_npi_lib_path or 'libNPI.so'}"
            )

    def get_signal(self, str signal) -> NpiFsdbSignal:
        """Look up a signal by its full hierarchical path and return an NpiFsdbSignal handle."""
        cdef npiFsdbSigHandle signal_handle = npi_fsdb_sig_by_name(self.fsdb_handle, signal.encode('ascii'), NULL)
        assert signal_handle != NULL, f"can't find signal: {signal}"
        return NpiFsdbSignal.init(signal_handle)

    @cython.boundscheck(False)  # 关闭边界检查以提升性能
    @cython.wraparound(False)   # 关闭负索引检查以提升性能
    def load_value_change(
        self,
        NpiFsdbSignal signal,
        unsigned long long begin_time,
        unsigned long long end_time,
        int xz_value
    ) -> np.ndarray:

        cdef int width = signal.width()
        cdef npiFsdbVctHandle signal_vct_handle = npi_fsdb_create_vct(signal.sig_handle)
        assert signal_vct_handle != NULL, f"can't create vct for signal"
        cdef npiFsdbTime cur_time
        cdef npiFsdbValue cur_value
        cdef int stat

        cdef vector[unsigned long long] time_array

        cdef int value_array_num = (width+63)//64
        assert width <= 64*64
        cdef vector[unsigned long long] value_array[64]

        cdef int first = 1
        cdef int i
        cdef unsigned long long chunk_values[64]
        while True:
            if first:
                stat = npi_fsdb_goto_time(signal_vct_handle, <npiFsdbTime> begin_time)
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

            cstr_to_ull_array(<char*>cur_value.value.str, xz_value, chunk_values, width)
            for i in range(value_array_num):
                value_array[i].push_back(chunk_values[i])
            first = False

        cdef np.ndarray[np.uint64_t, ndim=2] result_uint64 = np.zeros((value_array[0].size(), 2), dtype=np.uint64)
        cdef result_object = np.zeros((value_array[0].size(), 2), dtype=np.object_)
        cdef unsigned int shift

        npi_fsdb_release_vct(signal_vct_handle)

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

    def top_scope_list(self) -> list:
        cdef npiFsdbScopeIter top_scope_iter
        cdef npiFsdbScopeHandle top_scope_handle

        res = []
        top_scope_iter = npi_fsdb_iter_top_scope(self.fsdb_handle)
        while True:
            top_scope_handle = npi_fsdb_iter_scope_next(top_scope_iter)
            if top_scope_handle == NULL:
                break
            res.append(NpiFsdbScope.init(top_scope_handle))
        npi_fsdb_iter_scope_stop(top_scope_iter)
        return res

    def min_time(self) -> int:
        cdef npiFsdbTime time
        npi_fsdb_min_time(self.fsdb_handle, &time)
        return time

    def max_time(self) -> int:
        cdef npiFsdbTime time
        npi_fsdb_max_time(self.fsdb_handle, &time)
        return time

    def close(self):
        npi_fsdb_close(self.fsdb_handle)
