#pragma once

#include "../3dsetup/typedefs.h"
#include <cstddef>

namespace hetero_const {

static constexpr INDEX_TYPE QH = 19;

static constexpr INDEX_TYPE cx[19] = {
     0,  1, -1,  0,  0,  0,  0,  1, -1,  1,
    -1,  0,  0,  1, -1,  1, -1,  0,  0
};

static constexpr INDEX_TYPE cy[19] = {
     0,  0,  0,  1, -1,  0,  0,  1, -1,  0,
     0,  1, -1, -1,  1,  0,  0,  1, -1
};

static constexpr INDEX_TYPE cz[19] = {
     0,  0,  0,  0,  0,  1, -1,  0,  0,  1,
    -1,  1, -1,  0,  0, -1,  1, -1,  1
};

static constexpr VALUE_TYPE w[19] = {
    VALUE_TYPE(1.0/3.0),
    VALUE_TYPE(1.0/18.0), VALUE_TYPE(1.0/18.0),
    VALUE_TYPE(1.0/18.0), VALUE_TYPE(1.0/18.0),
    VALUE_TYPE(1.0/18.0), VALUE_TYPE(1.0/18.0),
    VALUE_TYPE(1.0/36.0), VALUE_TYPE(1.0/36.0),
    VALUE_TYPE(1.0/36.0), VALUE_TYPE(1.0/36.0),
    VALUE_TYPE(1.0/36.0), VALUE_TYPE(1.0/36.0),
    VALUE_TYPE(1.0/36.0), VALUE_TYPE(1.0/36.0),
    VALUE_TYPE(1.0/36.0), VALUE_TYPE(1.0/36.0),
    VALUE_TYPE(1.0/36.0), VALUE_TYPE(1.0/36.0)
};

static constexpr INDEX_TYPE D[19] = {
      19, 2394, 252, 10, 40, 10, 40, 10, 40, 36,
      72,   12,  24,  4,  4,  4,  8,  8,  8
};

inline void fill_s(VALUE_TYPE tau, INDEX_TYPE stype, VALUE_TYPE s[19]) {
    if (stype == 0) {
        s[0] = VALUE_TYPE(0.0);
        s[1] = VALUE_TYPE(1.64);
        s[2] = VALUE_TYPE(1.54);
        s[3] = VALUE_TYPE(0.0);
        s[4] = VALUE_TYPE(8.0) * (VALUE_TYPE(2.0) - VALUE_TYPE(1.0) / tau) /
               (VALUE_TYPE(8.0) - VALUE_TYPE(1.0) / tau);
        s[5] = VALUE_TYPE(0.0);
        s[6] = VALUE_TYPE(8.0) * (VALUE_TYPE(2.0) - VALUE_TYPE(1.0) / tau) /
               (VALUE_TYPE(8.0) - VALUE_TYPE(1.0) / tau);
        for (int i = 7; i < 19; i++) {
            s[i] = VALUE_TYPE(1.0) / tau;
        }
    } else {
        s[0] = VALUE_TYPE(1.0);
        s[1] = VALUE_TYPE(1.1);
        s[2] = VALUE_TYPE(1.0);
        s[3] = VALUE_TYPE(1.0);
        s[4] = VALUE_TYPE(1.2);
        s[5] = VALUE_TYPE(1.0);
        s[6] = VALUE_TYPE(1.2);
        for (int i = 7; i < 19; i++) {
            s[i] = VALUE_TYPE(1.0) / tau;
        }
    }
}

inline size_t scalar_index(INDEX_TYPE i, INDEX_TYPE j, INDEX_TYPE k,
                           INDEX_TYPE Nnx, INDEX_TYPE Nny) {
    return static_cast<size_t>(i) +
           static_cast<size_t>(j) * static_cast<size_t>(Nnx) +
           static_cast<size_t>(k) * static_cast<size_t>(Nnx) * static_cast<size_t>(Nny);
}

inline size_t pdf_index(INDEX_TYPE l, INDEX_TYPE i, INDEX_TYPE j, INDEX_TYPE k,
                        INDEX_TYPE Nnx, INDEX_TYPE Nny, INDEX_TYPE Nnz) {
    const size_t Ntot =
        static_cast<size_t>(Nnx) *
        static_cast<size_t>(Nny) *
        static_cast<size_t>(Nnz);

    return scalar_index(i, j, k, Nnx, Nny) +
           static_cast<size_t>(l) * Ntot;
}

} // namespace hetero_const
