//#####################################################################
// Copyright (c) 2017, Haixiang Liu
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __SPGrid_Blocks_h__
#define __SPGrid_Blocks_h__

#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <vector>
#include <SPGrid/Core/SPGrid_Mask.h>
#include <pthread.h>

namespace SPGrid{

//#####################################################################
// Class SPGrid_Blocks
//#####################################################################
template<int log2_struct,int D>
class SPGrid_Blocks
{
    pthread_mutex_t pm_mutex;
    typedef SPGrid_Mask_base<log2_struct,D> T_MASK;

public:
    unsigned long* page_mask_array;
    unsigned long array_length;
    unsigned long max_linear_offset; // TODO: Change semantics to make this the first offset that is *invalid*
    std::vector<unsigned long> block_offsets;
    bool dirty; // Indicates that block offsets are inconsistent with the bitmap (perhaps for a good reason, if only one of them is used)

    // Single size argument constructor    
    SPGrid_Blocks(unsigned long number_of_elements)
    {
        unsigned long number_of_pages = number_of_elements>>T_MASK::block_bits;
        array_length = (number_of_pages+0x3fUL)>>6;
        page_mask_array=new unsigned long[array_length]; // TODO: Why "new" and not Raw_Allocate() ?
        memset(reinterpret_cast<void*>(page_mask_array),0,array_length*sizeof(unsigned long)); // TODO: Is this really needed?
        max_linear_offset = number_of_elements*(1u<<T_MASK::data_bits); // TODO: Check this is correct
        pthread_mutex_init(&pm_mutex,0); // TODO: Check to see where these mutexes are really used
        dirty = false; // They start as consistent -- both empty
    }
    
    inline bool CheckBounds(unsigned long linear_offset)
    {
        return (linear_offset < max_linear_offset);
    }

    void MarkPageActive(unsigned long linear_offset)
    {
        if( linear_offset < max_linear_offset )
        {
            unsigned long page_mask = 1UL << (linear_offset>>12 & 0x3f);
            page_mask_array[linear_offset>>18] |= page_mask;
            dirty = true;
        } 
        else 
            FATAL_ERROR("Linear offset "+Value_To_String(linear_offset)+" is out of range (upper limit = "+Value_To_String(max_linear_offset)+")");
    }

    bool IsPageActive(const unsigned long linear_offset) const
    {
        if( linear_offset < max_linear_offset )
        {
            unsigned long page_mask = 1UL << (linear_offset>>12 & 0x3f);
            return page_mask_array[linear_offset>>18] & page_mask;
        }else
            return false;
    }

    std::pair<const unsigned long*,unsigned> Get_Blocks() const
    {
        if(block_offsets.size())
            return std::pair<const unsigned long*,unsigned>(&block_offsets[0],block_offsets.size());
        else
            return std::pair<const unsigned long*,unsigned>((const unsigned long*)0,0);
    }

    void Refresh_Block_Offsets()
    {
        if(dirty)
            block_offsets = GenerateBlockOffsets();
        dirty = false;
    }

    std::vector<unsigned long> GenerateBlockOffsets()
    {
        std::vector<unsigned long> block_offsets;
        for (unsigned long i = 0; i<array_length; i++)
        {
            if(page_mask_array[i])
            {
                for (unsigned long pos=0; pos<64; pos++)
                {
                    if(page_mask_array[i] & (1UL<<pos))
                        block_offsets.push_back((i<<18)|(pos<<12));
                }
            }
        }
        return block_offsets;
    }

    void Clear_Bitmap()
    {
        memset(reinterpret_cast<void*>(page_mask_array),0,array_length*sizeof(unsigned long));
        dirty=true;
    }

    void Clear_Blocks()
    {
        std::vector<unsigned long>().swap(block_offsets);
        dirty=true;
    }

    void Clear()
    {Clear_Bitmap();Clear_Blocks();dirty=false;}

    void FillBitmapWithBlockOffsets(std::vector<unsigned long> block_offsets)
    {
        dirty = true;
        for (std::vector<unsigned long>::iterator it = block_offsets.begin(); it != block_offsets.end(); ++it)
        {
            unsigned long cur_offset = *it;
            page_mask_array[cur_offset>>18] |= ( 1UL << (cur_offset>>12 & 0x3f) );
        }
    }
};
}
#endif
