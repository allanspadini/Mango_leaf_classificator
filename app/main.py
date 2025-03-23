import torch
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms

# Configurar o dispositivo (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o modelo completo
model = torch.load("mango_leaf.pth", map_location=device, weights_only=False)
model.to(device)
model.eval()

# Lista de classes
class_names = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould"
]

# Transformação da imagem (igual à usada no treinamento)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Função para prever a classe da imagem
def predict(image):
    image = image.convert("RGB")  # Garante que a imagem esteja no formato correto
    input_tensor = transform(image).unsqueeze(0).to(device)  # Adiciona batch dimension e move para GPU/CPU
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    return class_names[predicted.item()]  # Retorna a classe prevista

# Criar interface Gradio
def main():
    interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs="text",
        title="Classificação de Folhas de Manga",
        description="Faça o upload de uma imagem de folha de manga e receba a previsão do modelo."
    )
    interface.launch(share=True, server_name="0.0.0.0",server_port=8080)

if __name__ == "__main__":
    main()
