#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define MAGIC_NUM 0x414D414C

//For dataset outside of HF!

int main(){

    uint32_t vocab_size = 128256;
    uint32_t total_tokens = 500000; 
    uint32_t train_tokens = (uint32_t)(total_tokens * 0.90);
    uint32_t val_tokens = total_tokens - train_tokens;
    int32_t *tokens = (int32_t *)malloc(total_tokens * sizeof(int32_t));

    if(!tokens){
        printf("Error in memory alloc");
        return 1;
    }
    //Dummy data
    for (uint32_t i = 0; i < total_tokens; i++) {
        tokens[i] = rand() % vocab_size;
    }

    
    FILE *f_train = fopen("../data/train.bin", "wb");
    FILE *f_val = fopen("../data/val.bin", "wb");
    
    if(!f_train || !f_val){
        printf("Error in opening file");
        return 1;
    }

    uint32_t magic = MAGIC_NUM;

    fwrite(&magic, sizeof(uint32_t), 1, f_train);
    fwrite(&vocab_size, sizeof(uint32_t), 1, f_train);
    fwrite(&train_tokens, sizeof(uint32_t), 1, f_train);

    fwrite(tokens, sizeof(int32_t), train_tokens, f_train);

    fwrite(&magic, sizeof(uint32_t), 1, f_val);
    fwrite(&vocab_size, sizeof(uint32_t), 1, f_val);
    fwrite(&val_tokens, sizeof(uint32_t), 1, f_val);

    fwrite(tokens + train_tokens, sizeof(int32_t), val_tokens, f_val);

    printf("packing data complete!");
    return 0;
}