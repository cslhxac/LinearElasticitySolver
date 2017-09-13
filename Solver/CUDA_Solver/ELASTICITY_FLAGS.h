//#####################################################################
// Copyright (c) 2017, Haixiang Liu.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __ELASTICITY_FLAGS_H__
#define __ELASTICITY_FLAGS_H__

namespace SPGrid{
enum {
    // Node properties
    Elasticity_Node_Type_Active       = 0x00000001u,
    Elasticity_Node_Type_DirichletX   = 0x00000002u,
    Elasticity_Node_Type_DirichletY   = 0x00000004u,
    Elasticity_Node_Type_DirichletZ   = 0x00000008u,
};
}

#endif
