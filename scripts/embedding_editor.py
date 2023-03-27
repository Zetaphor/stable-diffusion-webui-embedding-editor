import datetime
import os
from webui import wrap_gradio_gpu_call
from modules import scripts, script_callbacks
from modules import shared, devices, sd_hijack, processing, sd_models, images, ui
from modules.shared import opts, cmd_opts, restricted_opts
from modules.ui import create_output_panel, setup_progressbar, create_refresh_button
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, \
    StableDiffusionProcessingImg2Img, process_images
from modules.ui import plaintext_to_html
from modules.textual_inversion.textual_inversion import save_embedding
import gradio as gr
import gradio.routes
import gradio.utils
import torch
import json
import traceback
import pickle

# ISSUES
# distribution shouldn't be fetched until the first embedding is opened, and can probably be converted into a numpy array
# most functions need to verify that an embedding is selected
# vector numbers aren't verified (might be better as a slider)
# weight slider values are lost when changing vector number
# remove unused imports
#
# TODO
# add tagged positions on sliders from user-supplied words (and unique symbols & colours)
# add a word->substrings printout for use with the above for words which map to multiple embeddings (e.g. "computer" = "compu" and "ter")
# add the ability to create embeddings which are a mix of other embeddings (with ratios), e.g. 0.5 * skunk + 0.5 * puppy is a valid embedding
# add the ability to shift all weights towards another embedding with a master slider
# add a strength slider (multiply all weights)
# print out the closest word(s) in the original embeddings list to the current embedding, with torch.abs(embedding1.vec - embedding2.vec).mean() or maybe sum
# also maybe print a mouseover or have an expandable per weight slider for the closest embedding(s) for that weight value
# maybe allowing per-weight notes, and possibly a way to save them per embedding vector
# add option to vary individual weights one at a time and geneerate outputs, potentially also combinations of weights. Potentially use scoring system to determine size of change (maybe latents or clip interrogator)
# add option to 'move' around current embedding position and generate outputs (a 768-dimensional vector spiral)?

embedding_editor_weight_visual_scalar = 1


def determine_embedding_distribution():
    cond_model = shared.sd_model.cond_stage_model
    embedding_layer = cond_model.wrapped.transformer.text_model.embeddings

    # fix for medvram/lowvram - can't figure out how to detect the device of the model in torch, so will try to guess from the web ui options
    device = devices.device
    if cmd_opts.medvram or cmd_opts.lowvram:
        device = torch.device("cpu")
    #

    for i in range(49405):  # guessing that's the range of CLIP tokens given that 49406 and 49407 are special tokens presumably appended to the end
        embedding = embedding_layer.token_embedding.wrapped(
            torch.LongTensor([i]).to(device)).squeeze(0)
        if i == 0:
            distribution_floor = embedding
            distribution_ceiling = embedding
        else:
            distribution_floor = torch.minimum(distribution_floor, embedding)
            distribution_ceiling = torch.maximum(
                distribution_ceiling, embedding)

    # a hack but don't know how else to get these values into gradio event functions, short of maybe caching them in an invisible gradio html element
    global embedding_editor_distribution_floor, embedding_editor_distribution_ceiling
    embedding_editor_distribution_floor = distribution_floor
    embedding_editor_distribution_ceiling = distribution_ceiling


def build_slider(index, default, weight_sliders):
    floor = embedding_editor_distribution_floor[index].item(
    ) * embedding_editor_weight_visual_scalar
    ceil = embedding_editor_distribution_ceiling[index].item(
    ) * embedding_editor_weight_visual_scalar

    slider = gr.Slider(minimum=floor, maximum=ceil, step="any",
                       label=f"w{index}", value=default, interactive=True, elem_id=f'embedding_editor_weight_slider_{index}')

    weight_sliders.append(slider)


def on_ui_tabs():
    determine_embedding_distribution()
    weight_sliders = []

    with gr.Blocks(analytics_enabled=False) as embedding_editor_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel', scale=1.5):
                with gr.Column():
                    with gr.Row():
                        embedding_name = gr.Dropdown(label='Embedding', elem_id="edit_embedding", choices=sorted(
                            sd_hijack.model_hijack.embedding_db.word_embeddings.keys()), interactive=True)
                        vector_num = gr.Number(
                            label='Vector', value=0, step=1, interactive=True)
                        refresh_embeddings_button = gr.Button(
                            value="Refresh Embeddings", variant='secondary')
                        save_embedding_button = gr.Button(
                            value="Save Embedding", variant='primary')

                    instructions = gr.HTML(f"""
                        <p>Enter words and color hexes to mark weights on the sliders for guidance. Hint: Use the txt2img prompt token counter or <a style="font-weight: bold;" href="https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer">webui-tokenizer</a> to see which words are constructed using multiple sub-words, e.g. 'computer' doesn't exist in stable diffusion's CLIP dictionary and instead 'compu' and 'ter' are used (1 word but 2 embedding vectors). Currently buggy and needs a moment to process before pressing the button. If it doesn't work after a moment, try adding a random space to refresh it.
                        </p>
                        """)
                    with gr.Row():
                        guidance_embeddings = gr.Textbox(value="apple:#FF0000, banana:#FECE26, strawberry:#FF00FF",
                                                         placeholder="symbol:color-hex, symbol:color-hex, ...", show_label=False, interactive=True)
                        guidance_update_button = gr.Button(
                            value='\U0001f504', elem_id='embedding_editor_refresh_guidance')
                        guidance_hidden_cache = gr.HTML(
                            value="", visible=False)

                        alignment_hidden_cache = gr.HTML(
                            value="", visible=False)

                        similarity_hidden_cache = gr.HTML(
                            value="", visible=False)

                    with gr.Column(elem_id='embedding_editor_weight_sliders_container'):
                        for i in range(0, 128):
                            with gr.Row():
                                build_slider(i*6+0, 0, weight_sliders)
                                build_slider(i*6+1, 0, weight_sliders)
                                build_slider(i*6+2, 0, weight_sliders)
                                build_slider(i*6+3, 0, weight_sliders)
                                build_slider(i*6+4, 0, weight_sliders)
                                build_slider(i*6+5, 0, weight_sliders)

            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    label='Output', show_label=False, elem_id="embedding_editor_gallery").style(grid=4)
                prompt = gr.Textbox(label="Prompt", elem_id=f"embedding_editor_prompt",
                                    show_label=False, lines=2, placeholder="e.g. A portrait photo of embedding_name")
                batch_count = gr.Slider(
                    minimum=1, step=1, label='Batch count', value=1)
                steps = gr.Slider(minimum=1, maximum=150,
                                  step=1, label="Sampling Steps", value=20)
                cfg_scale = gr.Slider(
                    minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0)
                seed = (gr.Textbox if cmd_opts.use_textbox_seed else gr.Number)(
                    label='Seed', value=-1)

                with gr.Row():
                    generate_preview = gr.Button(
                        value="Generate Preview", variant='primary')

                generation_info = gr.HTML()
                html_info = gr.HTML()

        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel', scale=1.5):
                with gr.Column():
                    with gr.Row():
                        input_align_to = gr.Textbox(elem_id="align_to_token", value="", label="Alignment Token",
                                                    placeholder="token", show_label=True, interactive=True)
                        btn_align_to_input = gr.Button(
                            value="Align To Token", variant='primary')
                    with gr.Row():
                        btn_find_similar_single = gr.Button(
                            value="Find Similar (Single)", variant='primary')
                        btn_find_similar_modular = gr.Button(
                            value="Find Similar (Modular)", variant='primary')
                        btn_generate_test = gr.Button(
                            value="Generate 0 Weight Test Image", variant='primary')
                        btn_build_distribution_pickle = gr.Button(
                            value="Create Weight Distribution Picke", variant='primary')
                        btn_pickle_index = gr.Button(
                            value="Create Weights Pickle", variant='primary')
                        btn_write_weights = gr.Button(
                            value="Create Weights JSON", variant='primary')

        preview_args = dict(
            fn=wrap_gradio_gpu_call(generate_embedding_preview),
            # _js="submit",
            inputs=[
                embedding_name,
                vector_num,
                prompt,
                steps,
                cfg_scale,
                seed,
                batch_count,
            ] + weight_sliders,
            outputs=[
                gallery,
                generation_info,
                html_info
            ],
            show_progress=True,
        )

        generate_preview_args = dict(fn=wrap_gradio_gpu_call(generate_test_images),
                                     inputs=[embedding_name, vector_num, prompt, steps,
                                             cfg_scale, seed, batch_count] + weight_sliders,
                                     outputs=[
            gallery,
            generation_info,
            html_info
        ], show_progress=True)

        generate_preview.click(**preview_args)

        btn_generate_test.click(**generate_preview_args)

        btn_align_to_input.click(
            fn=None,
            _js="align_to_embedding",
            inputs=[alignment_hidden_cache],
            outputs=[]
        )

        btn_write_weights.click(
            fn=write_token_weights,
        )

        def load_weights(*weights):
            pickled_weights = load_pickled_weights()
            return find_similar_modular(pickled_weights, weights)

        btn_find_similar_modular.click(
            fn=load_weights,
            inputs=[] + weight_sliders,
            outputs=[similarity_hidden_cache]
        )

        btn_find_similar_single.click(
            fn=find_similar_single,
            inputs=[] + weight_sliders,
            outputs=[similarity_hidden_cache]
        )

        btn_pickle_index.click(
            fn=build_index_pickle,
        )

        btn_build_distribution_pickle.click(
            fn=build_distribution_pickle,
        )

        selection_args = dict(
            fn=select_embedding,
            inputs=[
                embedding_name,
                vector_num,
            ],
            outputs=weight_sliders,
        )

        similarity_hidden_cache.change(
            fn=None,
            _js="update_similarities",
            inputs=[
                similarity_hidden_cache
            ]
        )

        embedding_name.change(**selection_args)
        vector_num.change(**selection_args)

        def refresh_embeddings():
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()  # refresh_method

            def refreshed_args(): return {"choices": sorted(
                sd_hijack.model_hijack.embedding_db.word_embeddings.keys())}  # refreshed_args
            args = refreshed_args() if callable(refreshed_args) else refreshed_args

            for k, v in args.items():
                setattr(embedding_name, k, v)

            return gr.update(**(args or {}))

        refresh_embeddings_button.click(
            fn=refresh_embeddings,
            inputs=[],
            outputs=[embedding_name]
        )

        save_embedding_button.click(
            fn=save_embedding_weights,
            inputs=[
                embedding_name,
                vector_num,
            ] + weight_sliders,
            outputs=[],
        )

        guidance_embeddings.change(
            fn=update_guidance_embeddings,
            inputs=[guidance_embeddings],
            outputs=[guidance_hidden_cache]
        )

        guidance_embeddings.change(
            fn=update_alignment_embeddings,
            inputs=[guidance_embeddings],
            outputs=[alignment_hidden_cache]
        )

        guidance_update_button.click(
            fn=None,
            _js="embedding_editor_update_guidance",
            inputs=[guidance_hidden_cache],
            outputs=[]
        )

        guidance_hidden_cache.value = update_guidance_embeddings(
            guidance_embeddings.value)

        alignment_hidden_cache.value = update_alignment_embeddings(
            guidance_embeddings.value)

    return [(embedding_editor_interface, "Embedding Editor", "embedding_editor_interface")]


def generate_test_images(embedding_name, vector_num, prompt, steps, cfg_scale, seed, batch_count, *weights):
    test_values = []

    for i in range(0, 128):
        for j in range(0, 6):
            index = i * 6 + j
            floor = embedding_editor_distribution_floor[index].item(
            ) * embedding_editor_weight_visual_scalar
            ceil = embedding_editor_distribution_ceiling[index].item(
            ) * embedding_editor_weight_visual_scalar
            test_values.append(0.0)

    return generate_embedding_preview(
        embedding_name, vector_num, prompt, steps, cfg_scale, seed, batch_count, *test_values)


def select_embedding(embedding_name, vector_num):
    embedding = sd_hijack.model_hijack.embedding_db.word_embeddings[embedding_name]
    vec = embedding.vec[int(vector_num)]
    weights = []

    for i in range(0, 768):
        weights.append(vec[i].item() * embedding_editor_weight_visual_scalar)

    return weights


def apply_slider_weights(embedding_name, vector_num, weights):
    embedding = sd_hijack.model_hijack.embedding_db.word_embeddings[embedding_name]
    vec = embedding.vec[int(vector_num)]
    old_weights = []

    for i in range(0, 768):
        old_weights.append(vec[i].item())
        vec[i] = weights[i] / embedding_editor_weight_visual_scalar

    return old_weights


def generate_embedding_preview(embedding_name, vector_num, prompt: str, steps: int, cfg_scale: float, seed: int, batch_count: int, *weights):
    old_weights = apply_slider_weights(embedding_name, vector_num, weights)

    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        seed=seed,
        steps=steps,
        cfg_scale=cfg_scale,
        n_iter=batch_count,
    )

    if cmd_opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    processed = process_images(p)

    p.close()

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    apply_slider_weights(embedding_name, vector_num, old_weights)  # restore

    return processed.images, generation_info_js, plaintext_to_html(processed.info)


def save_embedding_weights(embedding_name, vector_num, *weights):
    apply_slider_weights(embedding_name, vector_num, weights)
    embedding = sd_hijack.model_hijack.embedding_db.word_embeddings[embedding_name]
    checkpoint = sd_models.select_checkpoint()

    filename = os.path.join(
        shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')
    optimizer = torch.optim.AdamW([embedding.vec])

    save_embedding(embedding, optimizer, checkpoint,
                   embedding_name, filename, remove_cached_checksum=True)

    print(f"Saved embedding to {filename}")


def update_alignment_embeddings(text):
    try:
        cond_model = shared.sd_model.cond_stage_model
        embedding_layer = cond_model.wrapped.transformer.text_model.embeddings

        pairs = [x.strip() for x in text.split(',')]

        col_weights = {}

        for pair in pairs:
            word, col = pair.split(":")

            ids = cond_model.tokenizer(
                word, max_length=77, return_tensors="pt", add_special_tokens=False)["input_ids"]
            embedding = embedding_layer.token_embedding.wrapped(
                ids.to(devices.device)).squeeze(0)[0]
            weights = []

            for i in range(0, 768):
                weight = embedding[i].item()
                floor = embedding_editor_distribution_floor[i].item()
                ceiling = embedding_editor_distribution_ceiling[i].item()

                # adjust to range for using as a guidance marker along the slider
                # weight = (weight - floor) / (ceiling - floor)
                weights.append(weight)

            col_weights[word] = weights

        return col_weights
    except:
        return []


def update_guidance_embeddings(text):
    try:
        cond_model = shared.sd_model.cond_stage_model
        embedding_layer = cond_model.wrapped.transformer.text_model.embeddings

        pairs = [x.strip() for x in text.split(',')]

        col_weights = {}

        for pair in pairs:
            word, col = pair.split(":")

            ids = cond_model.tokenizer(
                word, max_length=77, return_tensors="pt", add_special_tokens=False)["input_ids"]
            embedding = embedding_layer.token_embedding.wrapped(
                ids.to(devices.device)).squeeze(0)[0]
            weights = []

            for i in range(0, 768):
                weight = embedding[i].item()
                floor = embedding_editor_distribution_floor[i].item()
                ceiling = embedding_editor_distribution_ceiling[i].item()

                # adjust to range for using as a guidance marker along the slider
                weight = (weight - floor) / (ceiling - floor)
                weights.append(weight)

            col_weights[col] = weights

        return col_weights
    except:
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


def build_distribution_pickle():
    # Find the min and max value for each weight
    # Move from the min value to the max value in the weight, increasing by a tiny amount
    # Record the top X weights for each step in the value
    # Final record should be a collection of ranges, from the min to the max, which contains the range of each unique collection of matches
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
    print('Finished loading pickled weights')

    weight_distribution = {}

    # Step threshold of 0.000001
    # I can go as low as 0.001
    # Index 0: [48931, 33233, 31146]
    # Index 69: [13416, 17893, 34605]
    # Index 259: [1075, 2115, 11135]
    # Index 700: [45781, 44323, 39564]
    # Around 68K calculations per 30 minutes

    # Current lowest threshold
    step_threshold = 0.001

    start_time = datetime.datetime.now()

    # Manually starting at higher range now!
    for i in range(0, 768):
        print(f"Building weight distribution for index {i}")

        floor = embedding_editor_distribution_floor[i].item()
        ceil = embedding_editor_distribution_ceiling[i].item()

        current_min_val = floor

        token_ranges = []

        # Started at 9:18PM

        # Initialize the list with the floor value
        found_tokens = closest_tokens(current_min_val, token_weights[i])

        # print(f"{i} initial list", found_tokens)

        # print(
        #     f"Index {i} total steps: {int(ceil / step_threshold) - int(floor / step_threshold)}")
        # print(
        #     f"Index {i} min/max : {floor}/{ceil}")

        time_diff = datetime.datetime.now() - start_time
        print(
            f"Starting index {i} at {round(time_diff.total_seconds(), 2)} seconds")

        for step_index in range(int(floor / step_threshold), int(ceil / step_threshold)):
            step_value = step_index * step_threshold
            step_tokens = closest_tokens(step_value, token_weights[i])
            # We're using sets here because it's probably fine if they're not in the same order if the values are the same
            if set(found_tokens) != set(step_tokens):
                token_ranges.append({
                    'min': current_min_val,
                    'max': step_value,
                    'tokens': step_tokens
                })
                current_min_val = step_value
                found_tokens = step_tokens
                # time_diff = datetime.datetime.now() - start_time
                # print(
                #     f"Added weight token range for step {step_value} in index {i} at {round(time_diff.total_seconds(), 2)} seconds")
                # print(step_tokens)
                # break  # Stop early for testing
        # break  # Stop early for testing

        weight_distribution[i] = token_ranges
        # Checkpoint the weight distribution

    time_diff = datetime.datetime.now() - start_time
    print(
        f'Saving weight distribution with index {i} at {round(time_diff.total_seconds(), 2)} seconds')
    with open(os.path.join(os.getcwd(), 'extensions/stable-diffusion-webui-embedding-editor/weights', sd_version + '-weight-distribution.pkl'), 'wb') as f:
        pickle.dump(weight_distribution, f)
    pickle_time_diff = datetime.datetime.now() - start_time
    print(
        f'Finished saving weight distribution at {round(pickle_time_diff.total_seconds(), 2)} seconds')
    print(weight_distribution)
    print('Finished building weight distribution!')


def closest_tokens(input_val, index_weights):
    input_tensor = torch.tensor([input_val])

    similarities = []
    top_x = 3

    for idx, token in enumerate(index_weights):
        token_tensor = torch.tensor([token])
        difference = torch.abs(input_tensor - token_tensor).item()
        if difference <= 0.1:
            similarities.append((difference, idx))

    # Sort by difference and take the top X
    top_x_similarities = sorted(
        similarities, key=lambda x: x[0], reverse=False)[:top_x]

    tokens_only = [tup[1] for tup in top_x_similarities]

    return tokens_only
# [(5.112588405609131e-05, 21757), (0.011545553803443909, 48931), (0.011637106537818909, 33233)]


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

    1929

    return []


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


script_callbacks.on_ui_tabs(on_ui_tabs)
