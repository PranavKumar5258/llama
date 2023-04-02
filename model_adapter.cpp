#include <cassert>
#include <cstring>
#include <fstream>
#include <regex>
#include <iostream>
#include <iterator>
#include <queue>
#include <string>
#include <math.h>
#include <vector>

#include "model_adapter.h"

static clock_t bench_timer = 0;

void timer_start()
{
    bench_timer = clock();
}
double timer_check()
{
    double ticks = clock() - bench_timer;
    double time_taken = ((double)ticks) / CLOCKS_PER_SEC;
    return time_taken;
}

void print_tok_vec(std::vector<int> &embd)
{
    std::cout << "[";
    bool first = true;
    for (auto i : embd)
    {
        if (!first)
        {
            std::cout << ',';
        }
        first = false;
        std::cout << i;
    }
    std::cout << "]";
}

//return val: 0=fail, 1=(original ggml, alpaca), 2=(ggmf), 3=(ggjt) 
 FileFormat check_file_format(const std::string & fname)
 {
    std::vector<char> f_buf(1024*1024);

    auto fin = std::ifstream(fname, std::ios::binary);
    fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return FileFormat::FAIL;
    }

    FileFormat fileformat = FileFormat::FAIL;
    uint32_t magic;
    fin.read((char *) &magic, sizeof(magic));
    if (magic == 0x67676d6c) {  //v1 format ggml, alpaca, old gptj and gpt2 models
       fileformat = FileFormat::GGML;
       //we need to read more to determine
       int32_t vocabsiz = 0;
       fin.read((char *) &vocabsiz, sizeof(int32_t));
       if(vocabsiz==50400) //know GPT-J vocab size
       {
           fileformat = FileFormat::GPTJ1;
       }
    }
    else if(magic == 0x67676d66) //v2 format ggmf
    {
        fileformat = FileFormat::GGHF;
    }
    else if(magic == 0x67676a74) //v3 format ggjt
    {
        fileformat = FileFormat::GGJT; //ggjt by default
    }
    fin.close();
    
    return fileformat;
 }