@llama.cpp
Feature: Parallel

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model file stories260K.gguf
    And   a model alias tinyllama-2
    And   42 as server seed
    And   32 KV cache size
    And   2 slots
    And   continuous batching
    Then  the server is starting
    Then  the server is healthy

  Scenario Outline: Multi users completion
    Given a prompt:
      """
      Write a very long story about AI.
      """
    And a prompt:
      """
      Write another very long music lyrics.
      """
    And <n_predict> max tokens to predict
    Given concurrent completion requests
    Then the server is busy
    Then the server is idle
    And  all slots are idle
    Then all prompts are predicted with <n_predict> tokens
    Examples:
      | n_predict |
      | 512       |

  Scenario Outline: Multi users OAI completions compatibility
    Given a system prompt You are a writer.
    And   a model tinyllama-2
    Given a prompt:
      """
      Write a very long book.
      """
    And a prompt:
      """
      Write another a poem.
      """
    And <n_predict> max tokens to predict
    And streaming is <streaming>
    Given concurrent OAI completions requests
    Then the server is busy
    Then the server is idle
    Then all prompts are predicted with <n_predict> tokens
    Examples:
      | streaming | n_predict |
      | disabled  | 512       |
      #| enabled   | 512       | FIXME: phymbert: need to investigate why in aiohttp with streaming only one token is generated
