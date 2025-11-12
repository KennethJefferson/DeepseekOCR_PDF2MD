"""
DeepSeek-OCR Service Wrapper
Handles model initialization and image processing
"""

import os
import torch
import logging
from typing import Optional, Dict, Any
import asyncio
from functools import lru_cache
import base64

try:
    from vllm import LLM, SamplingParams
    from vllm.multimodal import MultiModalData
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not available - will use alternative loading method")

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeepSeekOCRService:
    """
    Service wrapper for DeepSeek-OCR model
    Supports both vLLM and transformers backends
    """

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.85,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        backend: str = "auto"  # auto, vllm, transformers
    ):
        """
        Initialize the DeepSeek-OCR service

        Args:
            model_path: Path to the model weights
            gpu_memory_utilization: Fraction of GPU memory to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling (0 for deterministic)
            backend: Backend to use (auto, vllm, transformers)
        """
        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.backend = backend
        self.model = None
        self.tokenizer = None
        self.sampling_params = None

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model based on available backend"""
        # Determine backend
        if self.backend == "auto":
            if VLLM_AVAILABLE:
                self.backend = "vllm"
            elif TRANSFORMERS_AVAILABLE:
                self.backend = "transformers"
            else:
                raise RuntimeError("No backend available. Install vLLM or transformers.")

        logger.info(f"Initializing DeepSeek-OCR with backend: {self.backend}")

        if self.backend == "vllm":
            self._initialize_vllm()
        elif self.backend == "transformers":
            self._initialize_transformers()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        logger.info("Model initialization complete")

    def _initialize_vllm(self):
        """Initialize using vLLM backend"""
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed")

        try:
            # Initialize vLLM model
            self.model = LLM(
                model=self.model_path,
                enable_prefix_caching=False,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True,
                max_model_len=8192,  # DeepSeek-OCR supports up to 8K context
                tensor_parallel_size=1,  # Single GPU for now
                dtype="bfloat16",  # Use bfloat16 for better performance
                enforce_eager=True  # Disable CUDA graphs for stability
            )

            # Set sampling parameters
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                skip_special_tokens=False,
                stop=["<|im_end|>"]  # DeepSeek stop token
            )

            logger.info("vLLM model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {str(e)}")
            raise

    def _initialize_transformers(self):
        """Initialize using transformers backend"""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers is not installed")

        try:
            # Load model and tokenizer
            from transformers import AutoModelForVision2Seq, AutoProcessor

            logger.info(f"Loading model from {self.model_path}")

            # Load processor (includes tokenizer and image processor)
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # Load model
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()

            logger.info("Transformers model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize transformers: {str(e)}")
            raise

    async def process_image(self, image_path: str, prompt: str) -> str:
        """
        Process an image and generate markdown

        Args:
            image_path: Path to the image file
            prompt: Prompt for the model

        Returns:
            Generated markdown text
        """
        if self.backend == "vllm":
            return await self._process_image_vllm(image_path, prompt)
        else:
            return await self._process_image_transformers(image_path, prompt)

    async def _process_image_vllm(self, image_path: str, prompt: str) -> str:
        """Process image using vLLM backend"""
        try:
            # Read image file
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Encode image to base64 for vLLM
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            # Create messages in the format expected by DeepSeek-OCR
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

            # Generate response using async executor
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                self.model.generate,
                messages,
                self.sampling_params
            )

            # Extract generated text
            generated_text = outputs[0].outputs[0].text

            # Post-process the text (remove any remaining special tokens)
            generated_text = self._post_process_text(generated_text)

            return generated_text

        except Exception as e:
            logger.error(f"Error processing image with vLLM: {str(e)}")
            raise

    async def _process_image_transformers(self, image_path: str, prompt: str) -> str:
        """Process image using transformers backend"""
        try:
            from PIL import Image

            # Load image
            image = Image.open(image_path)

            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate response
            loop = asyncio.get_event_loop()
            with torch.no_grad():
                outputs = await loop.run_in_executor(
                    None,
                    lambda: self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature if self.temperature > 0 else None,
                        do_sample=self.temperature > 0,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id
                    )
                )

            # Decode the output
            generated_text = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True
            )[0]

            # Post-process
            generated_text = self._post_process_text(generated_text)

            return generated_text

        except Exception as e:
            logger.error(f"Error processing image with transformers: {str(e)}")
            raise

    def _post_process_text(self, text: str) -> str:
        """
        Post-process generated text

        Args:
            text: Generated text from the model

        Returns:
            Cleaned text
        """
        # Remove any remaining special tokens
        special_tokens = ["<|im_start|>", "<|im_end|>", "<|grounding|>", "<image>", "</image>"]
        for token in special_tokens:
            text = text.replace(token, "")

        # Clean up extra whitespace
        text = text.strip()

        # Ensure proper markdown formatting
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove leading/trailing whitespace from each line
            line = line.strip()
            if line:
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1]:  # Keep empty lines between content
                cleaned_lines.append('')

        return '\n'.join(cleaned_lines)

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the service

        Returns:
            Dictionary with health status information
        """
        status = {
            "status": "healthy" if self.model is not None else "unhealthy",
            "model_loaded": self.model is not None,
            "backend": self.backend,
            "cuda_available": torch.cuda.is_available(),
            "model_path": self.model_path
        }

        if torch.cuda.is_available():
            # Get GPU memory info
            gpu_mem_info = torch.cuda.mem_get_info()
            free_gb = gpu_mem_info[0] / 1e9
            total_gb = gpu_mem_info[1] / 1e9
            status["gpu_memory_free"] = f"{free_gb:.2f}GB"
            status["gpu_memory_total"] = f"{total_gb:.2f}GB"

            # Get GPU name
            status["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            status["gpu_memory_free"] = "N/A"
            status["gpu_memory_total"] = "N/A"
            status["gpu_name"] = "No GPU available"

        return status

    @lru_cache(maxsize=1)
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information (cached)

        Returns:
            Dictionary with model information
        """
        info = {
            "model_path": self.model_path,
            "backend": self.backend,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "gpu_memory_utilization": self.gpu_memory_utilization
        }

        if self.backend == "vllm" and hasattr(self.model, 'model_config'):
            info["model_config"] = {
                "max_model_len": self.model.model_config.max_model_len,
                "dtype": str(self.model.model_config.dtype)
            }

        return info

    def cleanup(self):
        """Cleanup resources"""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("OCR service cleanup complete")