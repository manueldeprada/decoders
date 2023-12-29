import torch
import numpy as np
import pytest
from decoders import BinaryCodeTransformer, BinaryCodeTransformerConfig  # Adjust the import according to your module structure

@pytest.fixture
def setup_model():
    torch.manual_seed(0)
    np.random.seed(0)
    config = BinaryCodeTransformerConfig(max_new_tokens=6)
    model = BinaryCodeTransformer(config)
    return model

# Each test is a function marked with pytest's test decorator
def test_binary_code_transformer_p(setup_model):
    model = setup_model
    input_ids = torch.tensor([[-1]] * 10)
    output = model.generate(input_ids, return_dict_in_generate=True, output_scores=True)
    
    expected_p = torch.tensor(
        [0.01277674, 0.02016235, 0.01482116, 0.01263750, 0.00884638, 0.01666631, 0.00923922, 0.03569583,
         0.05321665, 0.00776361, 0.02518661, 0.01208322, 0.01347602, 0.04171163, 0.00118292, 0.00146347,
         0.00032791, 0.02869582, 0.02417343, 0.03275462, 0.06173009, 0.02577006, 0.00993612, 0.02434604,
         0.00202075, 0.01639778, 0.00248399, 0.04646620, 0.01184487, 0.00859782, 0.00493299, 0.02389202,
         0.00977804, 0.01349049, 0.00030452, 0.01543374, 0.01520281, 0.01540431, 0.04620123, 0.01838373,
         0.00715223, 0.00922338, 0.01920197, 0.00099718, 0.01764163, 0.01782921, 0.00379200, 0.00221587,
         0.00608375, 0.00725792, 0.01355620, 0.00926820, 0.07151123, 0.00172794, 0.00376141, 0.00282407,
         0.01699675, 0.00468898, 0.01008079, 0.00449949, 0.00277934, 0.00187757, 0.01714653, 0.00238738],
        dtype=torch.float32
    )
    
    assert torch.allclose(model.p, expected_p), "model.p does not match expected values"

def test_binary_code_transformer_scores(setup_model):
    model = setup_model
    input_ids = torch.tensor([[-1]] * 10)
    output = model.generate(input_ids, return_dict_in_generate=True, output_scores=True)
    
    expected_scores = torch.tensor(
        [
            [-5.58296156, -5.86957741],  # each column is a different sample
            [-2.78498363, -3.65854216],
            [-2.63790083, -6.36082315],
            [-4.61157894, -3.71538639],
            [-4.18627453, -4.17310810],
            [-4.03749371, -4.02691698],
            [-4.43586063, -4.75624657],
            [-3.68144274, -4.41593790],
            [-4.30684376, -3.17697525],
            [-8.02278233, -3.55100393],
        ],
        dtype=torch.float32
    )
    
    assert torch.allclose(output.scores[5], expected_scores), "output.scores[5] does not match expected values"


if __name__ == "__main__":
    from arsenal import testing_framework
    testing_framework(globals())
