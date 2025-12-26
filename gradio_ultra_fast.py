import gradio as gr
import os
from pathlib import Path
import sys
import torch
from PIL import Image
import time
import gc
from contextlib import contextmanager

# Set up paths
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "run"))

# Change to run directory for correct relative paths
os.chdir(PROJECT_ROOT / "run")

# Optimize for aggressive memory management (4GB GPU)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['OMP_NUM_THREADS'] = '8'  # i7 typically has 8 threads
os.environ['MKL_NUM_THREADS'] = '8'

# Enable memory efficient attention if available
try:
    import xformers
    print("✅ xformers detected - memory efficient attention enabled")
except ImportError:
    print("⚠️  xformers not found - using standard attention (may use more VRAM)")

@contextmanager
def cuda_memory_manager():
    """Context manager for aggressive CUDA memory management"""
    torch.cuda.empty_cache()
    gc.collect()
    yield
    torch.cuda.empty_cache()
    gc.collect()

# Import after path setup
from utils_ootd import get_mask_location
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.aigc_run_parsing import Parsing
from ootd.inference_ootd_dc_low_vram import OOTDiffusionDC

# Initialize models (DC only for speed)
print("Initializing models...")
print("Optimized for: 16GB RAM, i7 CPU, 4GB GPU")
print("Target processing time: 8-12 minutes")

with cuda_memory_manager():
    openpose_model = OpenPose(0)
    parsing_model = Parsing(0)
    ootd_model = OOTDiffusionDC(0)
    
print("✅ Models loaded!")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

def process_ultra_fast(vton_img, garm_img, category, n_steps, image_scale, seed):
    """
    Ultra-fast processing optimized for 16GB RAM, i7 CPU, 4GB GPU
    """
    model_type = 'dc'
    
    # Validate inputs
    if vton_img is None or garm_img is None:
        return None, "❌ Error: Please upload both model and garment images"
    
    # Convert category
    if category == 'Upper-body':
        category_idx = 0
    elif category == 'Lower-body':
        category_idx = 1
    else:
        category_idx = 2

    start_time = time.time()
    
    try:
        with cuda_memory_manager():
            with torch.no_grad():
                # Load and resize images
                print("📸 Loading images...")
                garm_img = Image.open(garm_img).resize((768, 1024))
                vton_img = Image.open(vton_img).resize((768, 1024))
                
                # Process with OpenPose
                print("🏃 Running OpenPose...")
                keypoints = openpose_model(vton_img.resize((384, 512)))
                torch.cuda.empty_cache()  # Clear cache after each step
                
                # Human parsing
                print("👤 Running human parsing...")
                model_parse, _ = parsing_model(vton_img.resize((384, 512)))
                torch.cuda.empty_cache()

                # Generate mask
                print("🎭 Generating mask...")
                mask, mask_gray = get_mask_location(model_type, category_dict_utils[category_idx], model_parse, keypoints)
                mask = mask.resize((768, 1024), Image.NEAREST)
                mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
                
                masked_vton_img = Image.composite(mask_gray, vton_img, mask)
                torch.cuda.empty_cache()

                # Run diffusion
                print(f"🎨 Running diffusion with {n_steps} steps...")
                print(f"⚙️  Using {image_scale}x guidance scale")
                images = ootd_model(
                    model_type=model_type,
                    category=category_dict[category_idx],
                    image_garm=garm_img,
                    image_vton=masked_vton_img,
                    mask=mask,
                    image_ori=vton_img,
                    num_samples=1,  # Always 1 for speed
                    num_steps=n_steps,
                    image_scale=image_scale,
                    seed=seed,
                )

        elapsed_time = time.time() - start_time
        minutes = elapsed_time / 60
        print(f"✅ Completed in {elapsed_time:.1f} seconds ({minutes:.1f} minutes)")
        
        # Show VRAM usage
        if torch.cuda.is_available():
            vram_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"📊 Peak VRAM usage: {vram_used:.2f} GB")
        
        # Prepare mask images for display
        mask_images = [mask, mask_gray, masked_vton_img]
        
        return images, mask_images, f"✅ Completed in {minutes:.1f} minutes | Peak VRAM: {vram_used:.2f} GB"
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, f"❌ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="OOTDiffusion - Ultra Fast Mode", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ⚡ OOTDiffusion - Ultra Fast Mode
        ### Optimized for: 16GB RAM, i7 CPU, 4GB GPU
        ### Target time: 8-12 minutes per generation
        Upload your model image and garment, then click **Generate** to create a virtual try-on!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Model Image")
            vton_img = gr.Image(
                label="Upload Model Photo", 
                type="filepath", 
                height=400
            )
            
            # Check if example images exist
            example_path = os.path.join(PROJECT_ROOT, 'images')
            if os.path.exists(example_path):
                model_examples = [f for f in os.listdir(example_path) if f.startswith('model')]
                if model_examples:
                    gr.Examples(
                        examples=[[os.path.join(example_path, f)] for f in model_examples[:4]],
                        inputs=vton_img,
                        label="Example Models"
                    )
        
        with gr.Column(scale=1):
            gr.Markdown("### 👕 Garment Image")
            garm_img = gr.Image(
                label="Upload Garment Photo", 
                type="filepath", 
                height=400
            )
            
            # Check if example images exist
            if os.path.exists(example_path):
                cloth_examples = [f for f in os.listdir(example_path) if f.startswith('cloth')]
                if cloth_examples:
                    gr.Examples(
                        examples=[[os.path.join(example_path, f)] for f in cloth_examples[:4]],
                        inputs=garm_img,
                        label="Example Garments"
                    )
        
        with gr.Column(scale=1):
            gr.Markdown("### 🎨 Results")
            
            with gr.Tabs():
                with gr.TabItem("Generated Image"):
                    result_gallery = gr.Gallery(
                        label='Generated Images', 
                        show_label=False, 
                        elem_id="gallery", 
                        preview=True,
                        height=350,
                        columns=1
                    )
                
                with gr.TabItem("Mask Preview"):
                    mask_gallery = gr.Gallery(
                        label='Mask Images', 
                        show_label=False, 
                        elem_id="mask_gallery", 
                        preview=True,
                        height=350,
                        columns=2
                    )
            
            status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ⚙️ Settings")
            
            category = gr.Radio(
                choices=["Upper-body", "Lower-body", "Dress"],
                value="Dress",
                label="Garment Category",
                info="Select the type of garment"
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                n_steps = gr.Slider(
                    label="Diffusion Steps",
                    minimum=5,
                    maximum=25,
                    value=8,
                    step=1,
                    info="Lower = Faster, Higher = Better Quality (Recommended: 8-10 for 4GB GPU)"
                )
                
                image_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=3.0,
                    value=1.8,
                    step=0.1,
                    info="Controls how closely the result follows the garment (Recommended: 1.8 for 4GB GPU)"
                )
                
                seed = gr.Slider(
                    label="Seed",
                    minimum=-1,
                    maximum=2147483647,
                    step=1,
                    value=-1,
                    info="-1 for random seed"
                )
            
            run_button = gr.Button("🚀 Generate", variant="primary", size="lg")
    
    gr.Markdown(
        """
        ---
        ### 💡 Tips for Best Results:
        - Use clear, well-lit photos of the model (full body or upper body)
        - Garment images should be on a plain background
        - Match the garment category correctly (Upper-body/Lower-body/Dress)
        - First run takes longer due to CUDA compilation (~3-5 min extra)
        - Subsequent runs are faster!
        
        ### ⚡ Speed Presets (Optimized for 4GB GPU):
        - **Fast (8-10 min):** Steps: 8-10, Scale: 1.5-2.0
        - **Balanced (10-12 min):** Steps: 10-12, Scale: 2.0 ⭐ **Recommended**
        - **Quality (12-15 min):** Steps: 15-18, Scale: 2.0-2.5
        
        ### 🖥️ System Info:
        - **RAM:** 16GB (Excellent for preprocessing)
        - **CPU:** i7 (Fast image processing)
        - **GPU:** 4GB (Optimized with memory management)
        """
    )
    
    # Connect the button
    run_button.click(
        fn=process_ultra_fast,
        inputs=[vton_img, garm_img, category, n_steps, image_scale, seed],
        outputs=[result_gallery, mask_gallery, status_text]
    )

if __name__ == "__main__":
    print("=" * 60)
    print("Starting OOTDiffusion Ultra Fast UI")
    print("=" * 60)
    print("Opening browser at: http://localhost:7860")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    demo.launch(
        server_name='0.0.0.0',
        server_port=7860,
        share=False,
        show_error=True
    )
