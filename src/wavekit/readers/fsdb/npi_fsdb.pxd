ctypedef unsigned long long NPI_UINT64
ctypedef long long NPI_INT64
ctypedef int NPI_INT32
ctypedef unsigned int NPI_UINT32
ctypedef short NPI_INT16
ctypedef unsigned short NPI_UINT16
ctypedef char NPI_BYTE8
ctypedef unsigned char NPI_UBYTE8

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

ctypedef enum npiFsdbSigCompositeType_e:
    npiFsdbSigCtArray
    npiFsdbSigCtStruct
    npiFsdbSigCtUnion
    npiFsdbSigCtTaggedUnion
    npiFsdbSigCtRecord
