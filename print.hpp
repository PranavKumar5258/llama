//template<typename T> void print_fields(const T& obj);

#include <iostream>
//#include <refl.hpp>
#include "llama.h"

REFL_TYPE(ggml_init_params )
REFL_END

REFL_TYPE(ggml_opt_params::ggml_adam)
REFL_END

REFL_TYPE(ggml_opt_params::ggml_lbfgs)
REFL_END


REFL_TYPE(ggml_opt_context::ggml_grad )
REFL_END

REFL_TYPE(gpt_params )
REFL_END


REFL_TYPE(llama_sampling_context )
REFL_END

REFL_TYPE(llama_token_data )
REFL_END


REFL_TYPE(llama_token_data_array )
REFL_END

REFL_TYPE(llama_batch )
REFL_END


REFL_TYPE(ggml_object)
  REFL_FIELD(offs)
REFL_END

REFL_TYPE(ggml_tensor)
  REFL_FIELD(type)
REFL_END

REFL_TYPE(ggml_cplan)
  REFL_FIELD(work_size)
REFL_END

REFL_TYPE(ggml_hash_set)
  REFL_FIELD(size)
REFL_END

REFL_TYPE(ggml_cgraph)
  REFL_FIELD(size)
REFL_END

REFL_TYPE(ggml_scratch)
  REFL_FIELD(offs)
REFL_END

REFL_TYPE(ggml_compute_params)
  REFL_FIELD(type)
REFL_END

REFL_TYPE(ggml_opt_params)
  REFL_FIELD(type)
REFL_END

REFL_TYPE(ggml_opt_context)
  REFL_FIELD(ctx)
REFL_END

//REFL_TYPE(gguf_context)
//REFL_END

REFL_TYPE(gguf_init_params)
REFL_END

REFL_TYPE(ggml_something)
  REFL_FIELD(type_name)
REFL_END


//REFL_TYPE()
//  REFL_FIELD(d)
//REFL_TYPE()

// incomplete ttype
// REFL_TYPE(ggml_context)
//   REFL_FIELD(mem_size)
//   REFL_FIELD(mem_buffer)
// REFL_END

//REFL_TYPE(ggml_context_container)
//  REFL_FIELD(used)
//  REFL_FIELD(context)
//REFL_END

// REFL_TYPE(ggml_numa_node)
//   REFL_FIELD(cpus)
//   REFL_FIELD(n_cpus)
// REFL_END

// REFL_TYPE(ggml_numa_nodes)
//   REFL_FIELD(nodes)
//   REFL_FIELD(n_nodes)
// REFL_END

// REFL_TYPE(ggml_state)
//   REFL_FIELD(contexts)
//   REFL_FIELD(numa)
//   REFL_END

// REFL_TYPE(gguf_str)
//   REFL_FIELD(n)
//   REFL_FIELD(data)
// REFL_END

// REFL_TYPE(ggml_map_custom1_op_params)
//   REFL_FIELD(fun)
//   REFL_FIELD(n_tasks)
// REFL_END

// REFL_TYPE(ggml_map_custom2_op_params)
//   REFL_FIELD(fun)
//   REFL_FIELD(n_tasks)
// REFL_END

// REFL_TYPE(ggml_map_custom3_op_params)
//   REFL_FIELD(fun)
//   REFL_FIELD(n_tasks)
// REFL_END

// REFL_TYPE(hash_map)
//   REFL_FIELD(set)
//   REFL_FIELD(vals)
// REFL_END
// REFL_TYPE(ggml_compute_state_shared)
//   REFL_FIELD(cgraph)
//   REFL_FIELD(cplan)
// REFL_END
// REFL_TYPE(ggml_compute_state)
//   REFL_FIELD(thrd)
//   REFL_FIELD(ith)
// REFL_END
// REFL_TYPE(ggml_lbfgs_iteration_data)
//   REFL_FIELD(alpha)
//   REFL_FIELD(ys)
// REFL_END
//REFL_TYPE()
//  REFL_FIELD(type)
//REFL_END
// REFL_TYPE(gguf_kv)
//   REFL_FIELD(key)
//   REFL_FIELD(type)
// REFL_END

// REFL_TYPE(gguf_header)
//   REFL_FIELD(magic)
//   REFL_FIELD(version)
// REFL_END

// REFL_TYPE(gguf_tensor_info)
//   REFL_FIELD(name)
//   REFL_FIELD(n_dims)
// REFL_END

REFL_TYPE(gguf_context)
//  REFL_FIELD(header)
//  REFL_FIELD(kv)
REFL_END

// REFL_TYPE(gguf_buf)
//   REFL_FIELD(data)
//   REFL_FIELD(size)
// REFL_END

//REFL_TYPE(llama_token_data)
//REFL_END


REFL_TYPE(llama_model_params)
  REFL_FIELD(n_gpu_layers)
REFL_END
REFL_TYPE(llama_context_params)
  REFL_FIELD(seed)
REFL_END
REFL_TYPE(llama_model_quantize_params)
  REFL_FIELD(nthread)
REFL_END

REFL_TYPE(llama_grammar_element)
REFL_END

REFL_TYPE(llama_timings)
  REFL_FIELD(t_start_ms)
REFL_END
REFL_TYPE(llama_beam_view)
  REFL_FIELD(tokens)
REFL_END

REFL_TYPE(llama_beams_state)
  REFL_FIELD(beam_views)
REFL_END
  
//REFL_TYPE(ggml_backend)
//REFL_END

REFL_TYPE(ggml_backend_buffer)
REFL_END

//REFL_TYPE(ggml_allocr)
//REFL_END

//REFL_TYPE(ggml_tallocr)
//REFL_END

//REFL_TYPE(ggml_gallocr)
//REFL_END


//REFL_TYPE(llama_buffer)
//REFL_FIELD(data)
//REFL_FIELD(size)
//REFL_END
  

// REFL_TYPE(llama_file)
// REFL_FIELD(fp)
// REFL_FIELD(size)
// REFL_END
  

// REFL_TYPE(llama_mmap)
// REFL_FIELD(addr)
// REFL_FIELD(size)
// REFL_END


// REFL_TYPE(llama_mlock)
//   REFL_FIELD(addr)
//   REFL_FIELD(size)
// REFL_END

//REFL_TYPE(llama_state)
//  REFL_FIELD(log_callback)
//  REFL_FIELD(log_callback_user_data)
//  REFL_END
  

// REFL_TYPE(llama_hparams)
//   REFL_FIELD(vocab_only)
//   REFL_FIELD(n_vocab)
//   REFL_END


//REFL_TYPE(llama_cparams)
//  REFL_FIELD(n_ctx)
//  REFL_FIELD(n_batch)
//REFL_END

//REFL_TYPE(llama_layer)
//  REFL_FIELD(attn_norm)
//  REFL_FIELD(attn_norm_b)
//REFL_END

// REFL_TYPE(llama_kv_cell)
//   REFL_FIELD(pos)
//   REFL_FIELD(delta)
// REFL_END

// REFL_TYPE(llama_kv_cache)
//   REFL_FIELD(has_shift)
//   REFL_FIELD(head)
// REFL_END

// REFL_TYPE(llama_vocab)
// REFL_END

REFL_TYPE(llama_model)
//  REFL_FIELD(type)
//  REFL_FIELD(arch)
REFL_END

REFL_TYPE(llama_context)
REFL_END

// REFL_TYPE(llama_model_loader)
//   REFL_FIELD(n_kv)
//   REFL_FIELD(n_tensors)
// REFL_END

// REFL_TYPE(llm_build_context)
//   REFL_FIELD(model)
//   REFL_FIELD(hparams)
// REFL_END

// REFL_TYPE(llm_offload_trie)
// REFL_END

// REFL_TYPE(llm_symbol)
//   REFL_FIELD(prev)
// REFL_END

// REFL_TYPE(llm_bigram_spm)
// REFL_END

// REFL_TYPE(llm_tokenizer_spm)
// REFL_END

// REFL_TYPE(llm_bigram_bpe)
// REFL_END

// REFL_TYPE(llm_tokenizer_bpe)
// REFL_END
  

// REFL_TYPE(fragment_buffer_variant)
// REFL_END
  

// REFL_TYPE(llama_partial_utf8)
//   REFL_FIELD(value)
//   REFL_FIELD(n_remain)
// REFL_END
  

REFL_TYPE(llama_grammar)
//  REFL_FIELD(rules)
//  REFL_FIELD(stacks)
REFL_END
  

//REFL_TYPE(llama_grammar_candidate)
//  REFL_FIELD(index)
//  REFL_FIELD(code_points)
//REFL_END
  

// REFL_TYPE(llama_beam)
//   REFL_FIELD(tokens)
//   REFL_FIELD(p)
// REFL_END
  

// REFL_TYPE(llama_logit_info)
//   REFL_FIELD(logits)
//   REFL_FIELD(n_vocab)
// REFL_END

// REFL_TYPE(llama_beam_search_data)
//   REFL_FIELD(ctx)
//   REFL_FIELD(n_beams)
// REFL_END


// REFL_TYPE(quantize_state_internal)
//   REFL_FIELD(model)
//   REFL_FIELD(params)
// REFL_END

// REFL_TYPE(llama_data_context)
// REFL_END
  
// REFL_TYPE(llama_data_buffer_context)
//   REFL_FIELD(ptr)
// REFL_END

// REFL_TYPE(llama_data_file_context)
//   REFL_FIELD(file)
// REFL_END
  
// // A simple struct with some fields and a function
// // A custom attribute to mark some fields as hidden
struct hidden : refl::attr::usage::field {};

// // Another struct with some fields and a function, using the custom attribute
// struct Person {
//     std::string name;
//     int age;
//     [[hidden]] std::string password;
//     void say_hello() const {
//         std::cout << "Hello, I'm " << name << " and I'm " << age << " years old.\n";
//     }
// };

// // A generic function to print out the fields of any object
template<typename T>
void print_fields(const T& ) {
//     // Get the type descriptor of the object
  constexpr auto type = refl::reflect<T>();
  
//     // Print the type name
//  std::cout << "DEBUG:" << TypeName<T>.fullname_intern() << "\n";
  std::cout << "DEBUG Type: " << type.name.c_str() << "\n";

  //  T instance{};
  for_each(refl::reflect<T>().members, [&](auto member) {

    std::cout <<     member.name.str() << "\n";
      
  });

     refl::util::for_each(type.members, [&](auto member) {
//         // Check if the member is a field and not hidden
       //if ((refl::descriptor::is_field(member)) && (!member.has_attribute<hidden>()))) {
       //if ((refl::descriptor::is_field(member))) {
//             // Print the member name and value
	 std::cout << member.name << ": " << "\n";
	 //	 refl::get(member, obj)
	 //}
     });
     std::cout << "\n";
}

