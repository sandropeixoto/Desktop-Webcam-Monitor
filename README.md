# Desktop Webcam Monitor

Aplicação de monitoramento automático em Python com detecção de movimento e rostos.

## Funcionalidades
- **Detecção de Movimento:** Gatilho automático para gravação.
- **Identificação de Rostos:** Reconhecimento via MediaPipe.
- **Gravação Local:** Salva vídeos (.mp4) e snapshots (.jpg) na pasta `recordings/`.
- **Interface Visual:** Exibição em tempo real com marcações de detecção.

## Pré-requisitos
- Python 3.10+
- Webcam conectada

## Instalação
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Como Usar
1. Execute o monitor:
   ```bash
   python main.py
   ```
2. Teclas de comando:
   - `q`: Sair da aplicação.
   - `s`: Salvar um snapshot manual da imagem atual.

## Configuração
Edite o arquivo `config.py` para ajustar:
- `MOTION_THRESHOLD`: Sensibilidade do movimento.
- `AI_DETECTION_CONFIDENCE`: Precisão da detecção de rostos.
- `FPS`: Frames por segundo da gravação.
- `FRAME_WIDTH` / `FRAME_HEIGHT`: Resolução da câmera.

## Estrutura do Projeto
- `main.py`: Ponto de entrada e loop principal.
- `src/camera.py`: Gerenciamento da webcam.
- `src/detector.py`: Lógica de detecção (Movimento e IA).
- `src/recorder.py`: Serviço de gravação de arquivos.
- `recordings/`: Pasta onde os vídeos são salvos.
