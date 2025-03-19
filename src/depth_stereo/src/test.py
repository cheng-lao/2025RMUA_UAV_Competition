import torch
import torch.quantization
from depth2command.model import LSTMNetVIT_Modified

def remove_spectral_norm(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            try:
                torch.nn.utils.remove_spectral_norm(module)
            except ValueError:
                pass  # 如果没有应用 spectral_norm，则跳过
    return model

def convert_to_onnx(model, input_data, output_path):
    # 将 PyTorch 模型转换为 ONNX 模型
    torch.onnx.export(
        model, 
        [input_data], 
        output_path, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True, 
        input_names=['left', 'right', 'quantity', 'desired_vel'], 
        output_names=['output'], 
        dynamic_axes={'left': {0: 'batch_size'}, 'right': {0: 'batch_size'}, 'quantity': {0: 'batch_size'}, 'desired_vel': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        verbose=True  # 启用详细输出
    )

def calibrate_model(model, calibration_data):
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    # 使用校准数据运行模型，以便收集量化参数
    with torch.no_grad():
        model(calibration_data)

    torch.quantization.convert(model, inplace=True)
    return model

if __name__ == '__main__':
    # 加载 PyTorch 模型
    model = LSTMNetVIT_Modified()
    model.load_state_dict(torch.load('command_model_epoch116.pth'))
    model.eval()

    # 移除 spectral_norm
    model = remove_spectral_norm(model)

    # 创建示例输入
    x = torch.randn(1, 1, 720, 960)
    hidden_state = (torch.zeros(3, 128), torch.zeros(3, 128))
    # hidden_state = torch.zeros(1, 1, 720, 960)  # 使用零张量代替 None
    quantity = torch.randn(1, 4)
    desired_vel = torch.randn(1, 1)
    input_data = (x, desired_vel, quantity,hidden_state)

    calibration_data = input_data
    out, hidden_state = model(input_data)
    
    # 动态量化模型
    model = calibrate_model(model, calibration_data)

    # 转换为 ONNX 模型
    convert_to_onnx(model, input_data, 'model.onnx')