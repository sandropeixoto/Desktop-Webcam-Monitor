# Sugestões de Melhorias - Desktop Webcam Monitor

### 1. Interface Web
- Implementar **Streamlit** ou **Flask** para visualizar o feed e as gravações via navegador em outros dispositivos da rede local.

### 2. Notificações
- Integração com **Telegram Bot API** para enviar snapshots quando um rosto for detectado.
- Envio de alertas por e-mail com anexo.

### 3. Detecção Avançada
- Expandir o `AIDetector` para identificar objetos específicos (ex: animais, carros, pacotes) usando modelos **YOLO** ou **TensorFlow Lite**.

### 4. Gestão de Disco
- Implementar uma rotina de **auto-purge** que deleta gravações mais antigas que X dias ou quando o disco atingir X% de ocupação.

### 5. Multi-câmeras
- Refatorar o `main.py` para suportar múltiplas instâncias da classe `Camera` em threads separadas.
