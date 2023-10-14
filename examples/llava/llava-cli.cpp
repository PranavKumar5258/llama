#include <cstdio>
#include <cstdlib>

#include "ggml.h"
#include "common.h"
#include "clip.h"
#include "llava.h"
#include "llava-utils.h"


static void show_additional_info(int /*argc*/, char ** argv) {
    printf("\n example usage: %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    printf("  note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static bool load_image(llava_context * ctx_llava, gpt_params * params, float **image_embd, int * n_img_pos) {
    // load and preprocess the image
    clip_image_u8 img;
    auto prompt = params->prompt;
    if (prompt_contains_image(prompt)) {
        if (!params->image.empty()) {
            printf("using base64 encoded image instead of command line image path\n");
        }
        if (!clip_image_load_from_prompt(prompt, &img)) {
            fprintf(stderr, "%s: can't load image from prompt\n", __func__);
            return false;
        }
        prompt = remove_image_from_prompt(prompt);
    } else {
        if (!clip_image_load_from_file(params->image.c_str(), &img)) {
            fprintf(stderr, "%s: is %s really an image file?\n", __func__, params->image.c_str());
            return false;
        }
    }
    bool image_embed_result = llava_build_img_embed(ctx_llava->ctx_llama, ctx_llava->ctx_clip, params->n_threads, &img, image_embd, n_img_pos);
    if (!image_embed_result) {
        fprintf(stderr, "%s: coulnd't embed the image\n", __func__);
        return false;
    }

    return true;
}

static void process_prompt(struct llava_context * ctx_llava, float * image_embd, int n_img_pos, gpt_params * params, const char * prompt) {
    int n_past = 0;

    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;

    // llava chat format is "<system_prompt>USER: <image_embeddings>\n<textual_prompt>\nASSISTANT:"
    // GG: are we sure that the should be a trailing whitespace at the end of this string?
    eval_string(ctx_llava->ctx_llama, "A chat between a curious human and an artificial intelligence assistant.  The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER: ", params->n_batch, &n_past);
    printf("embedding image, n_img_pos is %d\n", n_img_pos);
    eval_image_embd(ctx_llava->ctx_llama, image_embd, n_img_pos, params->n_batch, &n_past);
    eval_string(ctx_llava->ctx_llama, prompt, params->n_batch, &n_past);
    eval_string(ctx_llava->ctx_llama, "\nASSISTANT:",        params->n_batch, &n_past);

    // generate the response

    printf("\n");

    for (int i = 0; i < max_tgt_len; i++) {
        const char * tmp = sample(ctx_llava->ctx_llama, *params, &n_past);
        if (strcmp(tmp, "</s>") == 0) break;

        printf("%s", tmp);
        fflush(stdout);
    }

    printf("\n");

}

int main(int argc, char ** argv) {
    ggml_time_init();

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        show_additional_info(argc, argv);
        return 1;
    }
    if (params.mmproj.empty() || (params.image.empty() && !prompt_contains_image(params.prompt))) {
        gpt_print_usage(argc, argv, params);
        show_additional_info(argc, argv);
        return 1;
    }

    auto ctx_llava = llava_init(&params);
    if (ctx_llava == NULL) {
        fprintf(stderr, "%s: error: failed to init llava\n", __func__);
        return 1;
    }

    float * image_embd;
    int n_image_pos;
    load_image(ctx_llava, &params, &image_embd, &n_image_pos);

    // process the prompt
    process_prompt(ctx_llava, image_embd, n_image_pos, &params, params.prompt.c_str());

    llama_print_timings(ctx_llava->ctx_llama);

    free(image_embd);
    llava_free(ctx_llava);
    return 0;
}
