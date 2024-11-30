
cdef extern from "npi.h":
    cdef int npi_init(int argc, char** argv)
    cdef int npi_end()
    cdef int npi_load_design(int argc, char** argv)

cdef extern from "npi_fsdb.h":

    ctypedef unsigned long long  NPI_UINT64
    ctypedef long long           NPI_INT64
    ctypedef int                 NPI_INT32
    ctypedef unsigned int        NPI_UINT32
    ctypedef short               NPI_INT16
    ctypedef unsigned short      NPI_UINT16
    ctypedef char                NPI_BYTE8
    ctypedef unsigned char       NPI_UBYTE8

    ctypedef void* npiFsdbFileHandle
    ctypedef void* npiFsdbScopeHandle
    ctypedef void* npiFsdbSigHandle
    ctypedef void* npiFsdbSigdbHandle
    ctypedef void* npiFsdbVctHandle
    ctypedef void* npiFsdbFtHandle

    ctypedef void* npiFsdbScopeIter
    ctypedef void* npiFsdbSigIter
    ctypedef void* npiFsdbEnumIter
    ctypedef NPI_UINT64 npiFsdbTime

    ctypedef enum npiFsdbValType:
        npiFsdbBinStrVal
        npiFsdbOctStrVal
        npiFsdbDecStrVal
        npiFsdbHexStrVal
        npiFsdbSintVal
        npiFsdbUintVal
        npiFsdbRealVal
        npiFsdbStringVal
        npiFsdbEnumStrVal
        npiFsdbSint64Val
        npiFsdbUint64Val
        npiFsdbObjTypeVal

    ctypedef struct npiFsdbValueValue:
        char* str
        int sint
        unsigned int uint
        long long sint64
        unsigned long long uint64
        double real

    ctypedef struct npiFsdbValue:
        npiFsdbValType format
        npiFsdbValueValue value
        # union field

    ctypedef enum npiFsdbSigPropertyType:
        npiFsdbSigName
        npiFsdbSigFullName
        npiFsdbSigIsReal
        npiFsdbSigHasMember
        npiFsdbSigLeftRange
        npiFsdbSigRightRange
        npiFsdbSigRangeSize
        npiFsdbSigIsString
        npiFsdbSigDirection
        npiFsdbSigAssertionType
        npiFsdbSigCompositeType
        npiFsdbSigIsPacked
        npiFsdbSigHasReasonCode
        npiFsdbSigReasonCode
        npiFsdbSigReasonCodeDesc
        npiFsdbSigIsParam
        npiFsdbSigHasEnum
        npiFsdbSigPowerType
        npiFsdbSigHasForceTag
        npiFsdbSigSpType
        npiFsdbSigEnumId

    ctypedef enum npiFsdbScopePropertyType:
        npiFsdbScopeName
        npiFsdbScopeFullName
        npiFsdbScopeDefName
        npiFsdbScopeType




    npiFsdbFileHandle npi_fsdb_open( const NPI_BYTE8* name )
    NPI_INT32 npi_fsdb_close( npiFsdbFileHandle file )
    NPI_INT32 npi_fsdb_min_time( npiFsdbFileHandle file, npiFsdbTime *time )
    NPI_INT32 npi_fsdb_max_time( npiFsdbFileHandle file, npiFsdbTime *time )
    npiFsdbSigHandle npi_fsdb_sig_by_name( npiFsdbFileHandle file, const NPI_BYTE8* name, npiFsdbScopeHandle scope )
    npiFsdbVctHandle npi_fsdb_create_vct( npiFsdbSigHandle sig )
    NPI_INT32 npi_fsdb_goto_time( npiFsdbVctHandle vct, npiFsdbTime time )
    NPI_INT32 npi_fsdb_goto_first( npiFsdbVctHandle vct )
    NPI_INT32 npi_fsdb_goto_next( npiFsdbVctHandle vct )
    NPI_INT32 npi_fsdb_goto_prev( npiFsdbVctHandle vct )

    NPI_INT32 npi_fsdb_vct_time( npiFsdbVctHandle vct, npiFsdbTime *time )
    NPI_INT32 npi_fsdb_vct_value( npiFsdbVctHandle vct, npiFsdbValue *value)
    NPI_INT32 npi_fsdb_release_vct( npiFsdbVctHandle vct )
    NPI_INT32 npi_fsdb_unload_vc( npiFsdbFileHandle file )

    NPI_INT32 npi_fsdb_sig_property( npiFsdbSigPropertyType type, npiFsdbSigHandle sig, NPI_INT32* prop )
    const NPI_BYTE8* npi_fsdb_sig_property_str( npiFsdbSigPropertyType type, npiFsdbSigHandle sig)
    NPI_INT32 npi_fsdb_scope_property( npiFsdbScopePropertyType type, npiFsdbScopeHandle scope, NPI_INT32* prop )
    const NPI_BYTE8* npi_fsdb_scope_property_str( npiFsdbScopePropertyType type, npiFsdbScopeHandle scope )

    npiFsdbScopeIter npi_fsdb_iter_top_scope( npiFsdbFileHandle file )
    npiFsdbScopeIter npi_fsdb_iter_child_scope( npiFsdbScopeHandle scope )
    npiFsdbScopeHandle npi_fsdb_iter_scope_next( npiFsdbScopeIter iter )
    npiFsdbSigHandle npi_fsdb_iter_sig_next( npiFsdbSigIter iter )
    npiFsdbSigIter npi_fsdb_iter_sig( npiFsdbScopeHandle scope )
    npiFsdbSigIter npi_fsdb_iter_member( npiFsdbSigHandle sig )
