def find_similar_single(*weights):
    embedder = shared.sd_model.cond_stage_model.wrapped
    sd_version = '1.x'
    if embedder.__class__.__name__ == 'FrozenCLIPEmbedder':  # SD1.x detected
        print('Using SD1.x embedder')

    elif embedder.__class__.__name__ == 'FrozenOpenCLIPEmbedder':  # SD2.0 detected
        print('Using SD2.x embedder')
        sd_version = '2.x'

    print('Loading pickled weights...')
    with open(os.path.join(os.getcwd(), 'extensions/stable-diffusion-webui-embedding-editor/weights', sd_version + '-weights.pkl'), 'rb') as f:
        token_weights = pickle.load(f)

    cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    input_tensor = torch.tensor([-0.00416564941406])
    # input_tensor = torch.tensor([token_weights[0][1929]])
    # print('Input tensor:', input_tensor)
    # print('Input tensor value:', input_tensor.item())

    similarities = []
    top_x = 3

    for idx, token in enumerate(token_weights[0]):
        token_tensor = torch.tensor([token])
        difference = torch.abs(input_tensor - token_tensor).item()
        if (difference <= 0.01):
            similarities.append((difference, idx))
        # print(f"{idx}: {difference}")

    # Sort by difference and take the top X
    top_x_similarities = sorted(similarities, key=lambda x: x[0])[:top_x]

    top_x_similarities = sorted(
        similarities, key=lambda x: x[0], reverse=False)[:top_x]

    print("Top", top_x, "most similar floats and their indices:", top_x_similarities)

    # print('Weight showing in slider:', weights[0])

    # print('Strawberry weight 0')
    # print(token_weights[0][10233])

    # print('Apple weight 0')
    # print(token_weights[0][3055])

    # print('Apple weight 0')
    # print(token_weights[0][8922])

    print('Dog weight 0')
    print(token_weights[0][1929])

    return []


def build_index_pickle():
    embedder = shared.sd_model.cond_stage_model.wrapped
    sd_version = '1.x'
    if embedder.__class__.__name__ == 'FrozenCLIPEmbedder':  # SD1.x detected
        print('Using SD1.x embedder')
        internal_embs = embedder.transformer.text_model.embeddings.token_embedding.wrapped.weight

    elif embedder.__class__.__name__ == 'FrozenOpenCLIPEmbedder':  # SD2.0 detected
        print('Using SD2.x embedder')
        sd_version = '2.x'
        internal_embs = embedder.model.token_embedding.wrapped.weight

    token_weights = {}
    for i in range(0, 768):
        token_weights[i] = []

    internal_len = len(internal_embs)

    start_time = datetime.datetime.now()

    print('Building weight index...')
    for i, row in enumerate(internal_embs):
        for j, col in enumerate(row):
            token_weights[j].append(col.item())
        if i % 1000 == 0:
            time_diff = datetime.datetime.now() - start_time
            print(
                f"Indexed {i} of {internal_len} at {round(time_diff.total_seconds(), 2)} seconds")

    print(
        f'Pickling weight index at {round(time_diff.total_seconds(), 2)} seconds')
    with open(os.path.join(os.getcwd(), 'extensions/stable-diffusion-webui-embedding-editor/weights', sd_version + '-weights.pkl'), 'wb') as f:
        pickle.dump(token_weights, f)
    print(
        f'Finished pickling weights at {round(time_diff.total_seconds(), 2)} seconds')


def load_pickled_weights():
    embedder = shared.sd_model.cond_stage_model.wrapped
    sd_version = '1.x'
    if embedder.__class__.__name__ == 'FrozenCLIPEmbedder':  # SD1.x detected
        print('Using SD1.x embedder')

    elif embedder.__class__.__name__ == 'FrozenOpenCLIPEmbedder':  # SD2.0 detected
        print('Using SD2.x embedder')
        sd_version = '2.x'

    loaded_weights = None
    print('Loading pickled weights...')
    with open(os.path.join(os.getcwd(), 'extensions/stable-diffusion-webui-embedding-editor/weights', sd_version + '-weights.pkl'), 'rb') as f:
        loaded_weights = pickle.load(f)
    print('Finished loading pickled weights')
    return loaded_weights


def find_similar_modular(pickled_weights, weights):
    output_ids = {}

    apple_0 = -0.0092010498046875
    apple_1 = 0.0009260177612304688
    print('Apple weight 0 valid:',
          pickled_weights[0][3055] == apple_0)
    print('Apple weight 0 valid:',
          pickled_weights[1][3055] == apple_1)

    strawberry_0 = -0.0011796951293945312
    strawberry_1 = -0.014251708984375
    print('Strawberry weight 0 valid:',
          pickled_weights[0][10233] == strawberry_0)
    print('Strawberry weight 1 valid:',
          pickled_weights[1][10233] == strawberry_1)

    print('Apple 0 Match:')
    matched = identify_similar_weight(pickled_weights, 0, apple_0)

    print('Apple 1 Match:')
    matched = identify_similar_weight(pickled_weights, 0, apple_1)

    print('Strawberry 0 Match:')
    matched = identify_similar_weight(pickled_weights, 0, strawberry_0)

    print('Strawberry 1 Match:')
    matched = identify_similar_weight(pickled_weights, 0, strawberry_1)

    return [output_ids]


def identify_similar_weight(pickled_weights, weight_index, weight_value):
    output_ids = {}
    similarities = []
    top_x = 5

    for token, token_value in enumerate(pickled_weights[weight_index]):
        # print('Input index:', weight_index)
        # print('Input token:', token)
        # print('Input tensor value:', pickled_weights[weight_index][token])
        input_tensor = torch.tensor(
            [pickled_weights[weight_index][token]])
        # print('Input tensor:', input_tensor)
        # print('Input tensor value:', input_tensor.item())
        # print(f"Weight {weight_index}, token {token}")

        token_tensor = torch.tensor([weight_value])
        difference = torch.abs(input_tensor - token_tensor).item()
        similarities.append((difference, token))

    top_x_similarities = sorted(
        similarities, key=lambda x: x[0], reverse=False)[:top_x]

    print(f"Weight {weight_index} top most similar floats and their indices:",
          top_x_similarities)

    # for idx, similarity in top_x_similarities:
    #     output_ids[idx] = similarity[1]

    return output_ids


def write_token_weights():
    try:
        embedder = shared.sd_model.cond_stage_model.wrapped
        if embedder.__class__.__name__ == 'FrozenCLIPEmbedder':  # SD1.x detected
            print('Using SD1.x embedder')
            internal_embs = embedder.transformer.text_model.embeddings.token_embedding.wrapped.weight

        elif embedder.__class__.__name__ == 'FrozenOpenCLIPEmbedder':  # SD2.0 detected
            print('Using SD2.x embedder')
            internal_embs = embedder.model.token_embedding.wrapped.weight

        vocab_weights = {}
        for i in range(0, 768):
            vocab_weights[i] = {}

        filename = 'sd1.5-vocab'
        file_dir = os.path.join(
            os.getcwd(), 'extensions/stable-diffusion-webui-embedding-editor/vocabs', filename + '.json')
        with open(file_dir, 'r') as f:
            data = json.load(f)
            data_len = len(data)

        start_time = datetime.datetime.now()

        print('Calculating vocab weights...')
        for i, (key, value) in enumerate(data.items()):
            emb_vec = internal_embs[value].squeeze(0)

            # Store the token weight in the following structure: weight number -> token_id -> weight value
            for j in range(0, 768):
                vocab_weights[j][value] = emb_vec[j].item()

            # Print every 1000 tokens
            if i % 1000 == 0:
                time_diff = datetime.datetime.now() - start_time
                print(
                    f"Stored {i} of {data_len} at {round(time_diff.total_seconds(), 2)} seconds")

        print('Saving vocab weights...')
        with open(os.path.join(os.getcwd(), 'extensions/stable-diffusion-webui-embedding-editor/weights', filename + '-weights.json'), 'w') as f:
            json.dump(vocab_weights, f)

        print('Saved vocab weights to', filename + '-weights.json')

    except Exception as e:
        print(e)
        traceback.print_exc()
        return []
