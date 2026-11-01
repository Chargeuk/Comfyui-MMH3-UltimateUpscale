"""CPU-only check; set COMFYUI_PATH to an installed ComfyUI source tree."""
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

sys.argv.append('--cpu')
sys.path.insert(0, os.environ['COMFYUI_PATH'])
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.nested_tensor
import comfy.model_management

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('mmh3_test', os.path.join(root, '__init__.py'), submodule_search_locations=[root])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from mmh3_test.nodes import nodes
sys.argv.remove('--cpu')

class EmbeddedOffloadTests(unittest.TestCase):
    def test_standalone_offloads_but_embedded_call_does_not(self):
        video = torch.zeros(1, 24, 2, 2, 2)
        audio = torch.zeros(1, 32, 2, 8)
        latent = {'samples': comfy.nested_tensor.NestedTensor((video, audio))}
        class Model:
            clone_base_uuid = 'test'
        model = Model()
        param = {'model_name': 'test', 'device': 'cuda', 'offload_model': True}
        def fake_upscale(v, p):
            return v, None, None
        def fake_sample(piece, *args):
            return piece['samples']
        with patch.object(nodes, 'upscale_latent', fake_upscale), \
             patch.object(nodes, 'sample_piece', fake_sample), \
             patch.object(comfy.model_management, 'unload_model_and_clones') as unload, \
             patch.object(comfy.model_management, 'soft_empty_cache'):
            args = dict(latent=latent, conditioning=[], model=model, noise=None,
                        sampler=None, sigmas=None, latent_upscale_param=param)
            nodes.MMH3UltimateUpscale.execute(**args)
            self.assertEqual(unload.call_count, 2)
            unload.reset_mock()
            nodes.MMH3UltimateUpscale.execute(**args, offload_diffusion=False)
            unload.assert_not_called()

if __name__ == '__main__':
    unittest.main()
