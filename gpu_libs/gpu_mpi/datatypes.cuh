#ifndef DATATYPES_CUH
#define DATATYPES_CUH

#include <stdbool.h>
#include <stdint.h>
#include <cassert>
// #include <list>

#include "mpi_types_list.cuh"

#define MPI_TYPES_ENUM_F(name, type) name
#define MPI_TYPES_ENUM_SEP ,

enum MPI_Datatype_Basic {
    MPI_TYPES_LIST(MPI_TYPES_ENUM_F, MPI_TYPES_ENUM_SEP)
};

#undef MPI_TYPES_ENUM_F
#undef MPI_TYPES_ENUM_SEP

struct MPI_Datatype;
// add a declaration before definition, because we have a circular reference Datatype_Ext -> Datatype -> Datatype_Ext

namespace gpu_mpi {


namespace detail {
    
template <MPI_Datatype_Basic> struct GetPlainType {};

#define MPI_TYPES_CONV_F(NAME, TYPE) template <> struct GetPlainType<NAME> { using type = TYPE; };
#define MPI_TYPES_CONV_SEP

MPI_TYPES_LIST(MPI_TYPES_CONV_F, MPI_TYPES_CONV_SEP)

#undef MPI_TYPES_CONV_F
#undef MPI_TYPES_CONV_SEP

} // namespace

template <MPI_Datatype_Basic T>
using PlainType = typename detail::GetPlainType<T>::type;

__device__ int plainTypeSize(MPI_Datatype_Basic type);
__device__ int TypeSize(MPI_Datatype type);

}// namespace

#undef MPI_TYPES_LIST

// const int TYPENAME_MAXLEN = 32;
const int TYPEMAP_MAXLEN = 101;

struct __Typemap_Tree {
    // tree-shape optimized storage
    // https://docs.google.com/presentation/d/1uSMdr3lLJNXwxSlTrT2pTTgP3028n8UEFWf_nRNpaN0/edit#slide=id.g1047b1fc64a_0_0

    size_t _viewlen_u, _size_u;  // extent of a single unit of the datatype represented by *this and all its subtrees (i.e., total extent divided by mul). doesn't change after creation
    int mul_o_typ;
    // If it's a leaf node, this variable means the basic type it refers to (or -1 for a gap); otherwise means multiplicity (as in the slides)
    // Thanks to the fact that we never need to attach any node to leaf nodes or delete any leaf nodes,
    //  there's no complex conversions among these two cases
    __Typemap_Tree *lch, *rch;  // left child right sibling

    __device__ __Typemap_Tree(int m): _size_u(0), _viewlen_u(0), mul_o_typ(m), lch(nullptr), rch(nullptr)
    {
        assert(m > -2);  // for all legitimate values, mul_o_typ can be -1 or an enum (>0)
    }  
    __device__ __Typemap_Tree(MPI_Datatype_Basic in_type): mul_o_typ(in_type),
        lch(nullptr), rch(nullptr), _size_u(gpu_mpi::plainTypeSize(in_type)), _viewlen_u(1) {}
    
    __device__ bool is_leaf() const {return lch == nullptr; }  // && rch == nullptr;}
    __device__ size_t size() const {return (is_leaf() ? 1 : mul_o_typ) * _size_u;}
    __device__ size_t viewlen() const {return (is_leaf() ? 1 : mul_o_typ) * _viewlen_u;}
    // __device__ friend bool operator == (const __Typemap_Tree a, const __Typemap_Tree b)
    // {
    //     // Problematic
    //     //  1
    //     //  |   should equal to  INT
    //     // INT
    //
    //     // through a PreOrder traversal
    //     if (a.mul_o_typ != b.mul_o_typ) return false;
    //     if ((a.lch == NULL) ^ (b.lch == NULL)) return false;
    //     if ((a.rch == NULL) ^ (b.rch == NULL)) return false;
    //     return (a.lch == b.lch || *(a.lch) == *(b.lch)) && (a.rch == b.rch || *(a.rch) == *(b.rch));
    // }
    __device__ friend bool operator == (const __Typemap_Tree a, const MPI_Datatype_Basic b)
    {
        if (a.lch || a.rch) return false;  // rch: this comparison cannot be called at non-root nodes
        return a.mul_o_typ == b;
    }
    __device__ bool type_check(MPI_Datatype_Basic etype) const
    {
        if (is_leaf())
        {
            if (mul_o_typ > 0) return mul_o_typ == etype;
            return true;  // mul_o_typ == -1 (gaps)
        }
        __Typemap_Tree* o = lch;
        while (o != nullptr)
        {
            if (o -> type_check(etype) == false) return false;
            o = o -> rch;
        }
        return true;
    }

    __device__ int __view_pos_to_file_pos(int view_pos) const
    {
        assert(view_pos <= viewlen());
        if (is_leaf()) return view_pos * _size_u;  // view_pos / _viewlen_u * _size_u: at leaf nodes, _viewlen_u must be 1

        int ret_mul = view_pos / _viewlen_u;
        view_pos %= _viewlen_u;
        if (view_pos == 0) return ret_mul * _size_u;

        __Typemap_Tree* o = lch;
        while (o != nullptr)
        {
            if (view_pos <= o->viewlen()) return o->__view_pos_to_file_pos(view_pos) + ret_mul * _size_u;
            view_pos -= o->viewlen(); o = o->rch;
        }
        assert(false);
        return -1;  // eliminate compile warnings
    }
    __device__ int __file_pos_to_view_pos(int file_pos) const
    {
        assert(file_pos <= size());
        if (is_leaf()) return file_pos / _size_u;  // file_pos / _size_u * _viewlen_u: at leaf nodes, _viewlen_u must be 1

        int ret_mul = file_pos / _size_u;
        file_pos %= _size_u;
        if (file_pos == 0) return ret_mul * _viewlen_u;

        __Typemap_Tree* o = lch;
        while (o != nullptr)
        {
            if (file_pos <= o->size()) return o->__file_pos_to_view_pos(file_pos) + ret_mul * _viewlen_u;
            file_pos -= o->size(); o = o->rch;
        }
        assert(false);
        return -1;
    }
};

struct MPI_Datatype {  // MPI datatype in extended format, for customized datatypes
    // For now, only two use cases are supported:
    // 1. get size of Datatype/Datatype_ext
    // 2. Compare Datatype_ext to Datatype so that we can simplify our code in special cases
    //    (comparing Datatype_ext to Datatype_ext is not needed atm)

    // int id;
    // size_t _size;  // extent of this datatype (including padding spaces)
    // char name[TYPENAME_MAXLEN];
    bool committed;
    // struct pii {MPI_Datatype_Basic basic_type;  int disp;} typemap[TYPEMAP_MAXLEN];
    // int typemap_len;
    // int typemap_gap;
    __Typemap_Tree* root;

    __device__ MPI_Datatype(): committed(false), root(nullptr) {}  // _size(0), typemap_len(0), typemap_gap(0) {}
    __device__ MPI_Datatype(MPI_Datatype_Basic in_type): committed(true), root(new __Typemap_Tree(in_type))  // _size(gpu_mpi::plainTypeSize(in_type)), typemap_len(1), typemap_gap(0)
    {
        // typemap[0].basic_type = in_type;
        // typemap[0].disp = 0;
    }  // for porting to MPI_Datatype_Basic
    __device__ operator MPI_Datatype_Basic() const
    {
        // assert(typemap_len==1);
        // assert(typemap_gap==0);
        // return MPI_Datatype_Basic(typemap[0].basic_type);
        assert(root != nullptr);
        if (root->is_leaf()) return (MPI_Datatype_Basic)(root -> mul_o_typ);
        if (root -> lch -> is_leaf() && root -> lch -> rch == nullptr && root -> rch == nullptr && root -> mul_o_typ == 1) return (MPI_Datatype_Basic)(root -> lch -> mul_o_typ);
        assert(false);
        return (MPI_Datatype_Basic)0;  // eliminate compile warnings
    } //- return value type does not match the function type

    __device__ size_t size() const {return root -> size();}  // {return _size;}
    __device__ size_t viewlen() const {return root -> viewlen();}
    __device__ size_t add_typemap_at_end(const MPI_Datatype rhs, int gap)
    {
        // assert(etype.typemap_len + typemap_len <= TYPEMAP_MAXLEN);

        // // if is the first element, ignore gap, i.e. the first disp in typemap is always 0
        // int start = typemap_len == 0 ? 0 : gap + typemap[typemap_len-1].disp + gpu_mpi::plainTypeSize(typemap[typemap_len-1].basic_type);
        // for (int i = 0; i < etype.typemap_len; i++)
        // {
        //     typemap[typemap_len + i].basic_type = (etype.typemap)[i].basic_type;
        //     typemap[typemap_len + i].disp       = start + (etype.typemap)[i].disp;
        // }
        // _size += _size == 0 ? etype.size() : etype.size() + gap;
        // typemap_len += etype.typemap_len;

        // return _size;

        assert(root != nullptr);

        add_gap(gap);

        root -> lch -> rch -> rch = rhs.root;
        root -> _size_u += rhs.size(); root -> _viewlen_u += rhs.viewlen();

        return size();
    }
    __device__ size_t self_replicate(int times)
    {
        assert(root != nullptr);

        if (root->is_leaf())
        {
            __Typemap_Tree* new_root = new __Typemap_Tree(1);
            // new_root -> mul_o_typ = 1;
            new_root -> lch = root;
            new_root -> _size_u = size(); new_root -> _viewlen_u = viewlen();
            root = new_root;
        }

        root -> mul_o_typ *= times;
        return size();
    }
    __device__ size_t add_gap(int gap)
    {
        // TODO: if multiplicity of root is 1, can directly insert to the rightest child instead of adding an additional top node

        assert(root != nullptr);

        __Typemap_Tree* new_root = new __Typemap_Tree(1);
        // new_root -> mul_o_typ = 1;
        new_root -> lch = root;

        __Typemap_Tree* gap_node = new __Typemap_Tree(-1);
        // gap_node -> mul_o_typ = -1;
        gap_node -> _size_u = gap;
        root -> rch = gap_node;  // as sibling
        new_root -> _size_u = size() + gap; new_root -> _viewlen_u = viewlen();

        root = new_root;
        return size();
    }

    __device__ bool type_check(MPI_Datatype etype) const
    {
        assert(root != nullptr);
        return root -> type_check((MPI_Datatype_Basic)etype);
    }
    // __device__ friend bool operator == (const MPI_Datatype a, const MPI_Datatype_Basic b)
    // {
    //     // Obsolete. Use the MPI_Datatype_Basic() above
    //     return a.root != nullptr && *(a.root) == b;
    // }
    // __device__ friend void copy_typemap_once(MPI_Datatype* newtype, const MPI_Datatype oldtype, int mul)  // obsolete
    // {
    //     int o_typemap_len = oldtype.typemap_len;
    //     for (int i = 0; i < o_typemap_len; i++)
    //     {
    //         (newtype -> typemap)[mul * o_typemap_len + i].basic_type = (oldtype.typemap)[i].basic_type;
    //         (newtype -> typemap)[mul * o_typemap_len + i].disp       = (oldtype.typemap)[i].disp + mul * oldtype.size();
    //     }
    //     _size = 
    // }

    __device__ int __view_pos_to_file_pos(int view_pos) const
    {
        view_pos += 1;  // from external, view_pos starts from 0; but in the tree, it starts from 1
        assert(root != nullptr);
        int ret_mul = view_pos / viewlen();
        view_pos %= viewlen();
        return root -> __view_pos_to_file_pos(view_pos) + ret_mul * size() - 1;
    }
    __device__ int __file_pos_to_view_pos(int file_pos) const
    {
        file_pos += 1;
        assert(root != nullptr);
        int ret_mul = file_pos / size();
        file_pos %= size();
        return root -> __file_pos_to_view_pos(file_pos) + ret_mul * viewlen() - 1;
    }
};

#endif
