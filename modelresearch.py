import os
import tensorflow as tf
from tensorflow import keras




def test_model(model: keras.Model, model_name: str):
    print(f"\n模型名称:{model_name}")
    save_dir = os.path.join('model_test',model_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    _network_summary = ""

    def pfn(s):
        nonlocal _network_summary
        _network_summary += (s + '\n')
    model.summary(print_fn=pfn)
    modelSummaryFP = os.path.join(save_dir,'summary.txt')
    with open(modelSummaryFP, 'w') as f:
        print(f"写入_network_summary：{modelSummaryFP}")
        f.write(_network_summary)

    modelJsonFP = os.path.join(save_dir,'model2json.json')
    with open(modelJsonFP, 'w') as f:
        print(f"写入model.to_json()：{modelJsonFP}")
        f.write(model.to_json())

    modelPlotFP = os.path.join(save_dir,'modelPlot.png')
    print(f"画图tf.keras.utils.plot_model：{modelPlotFP}")
    tf.keras.utils.plot_model(
        model,
        show_shapes=True,
        to_file=modelPlotFP,
        # expand_nested=True,
        dpi=100
    )


if __name__ == '__main__':
    model_dict = {
        'MobileNetV2': keras.applications.MobileNetV2,
        'MobileNetV3Small': keras.applications.MobileNetV3Small,
        'MobileNetV3Large': keras.applications.MobileNetV3Large,
        'EfficientNetB0': keras.applications.EfficientNetB0,
    }
    for model_name, model_func in model_dict.items():
        test_model(model=model_func(), model_name=model_name)
