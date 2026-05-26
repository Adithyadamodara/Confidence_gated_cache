import chainlit as cl
from main import handle_request, model

# Default configuration for the confidence-gated cache
config = {
    "confidence_method": "geometric_mean",
    "cache_threshold": 0.8,
    "serve_threshold": 0.5,
    "lambda": 0.01
}

@cl.on_chat_start
async def start():
    # Setup model settings in the sidebar
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="Model",
                label="Ollama Model",
                values=["gemma4:latest", "deepseek-r1:1.5b", "Mock Model"],
                initial_index=1,
            )
        ]
    ).send()
    
    # Ensure Ollama is active if not Mock
    model.use_ollama = True
    model.model_name = "deepseek-r1:1.5b"
    await cl.Message(content="Welcome to the Confidence-Gated Cache Interface!\nUse the **Settings** menu to change your model.").send()

@cl.on_settings_update
async def setup_agent(settings):
    selected_model = settings["Model"]
    if selected_model == "Mock Model":
        model.use_ollama = False
    else:
        model.use_ollama = True
        model.model_name = selected_model
        
    await cl.Message(content=f"⚙️ Switched to: **{selected_model}**").send()

@cl.on_message
async def main(message: cl.Message):
    # Display an empty message to show loading state
    msg = cl.Message(content="")
    await msg.send()
    
    # Process the query through the confidence-gated cache
    response_data = handle_request(message.content, config)
    
    output = response_data.get("output", "")
    cache_hit = response_data.get("cache_hit", False)
    confidence = response_data.get("confidence", 0.0)
    ttl = response_data.get("ttl", 0)
    
    # Construct a beautiful metadata badge string
    metadata = "### 📊 Request Analytics\n\n"
    if cache_hit:
        metadata += "- ⚡ **Status**: Cache Hit (Returned instantly)\n"
    else:
        metadata += "- 🧠 **Status**: Model Inference (Cache Miss)\n"
        
    metadata += f"- 🎯 **Confidence Score**: `{confidence:.4f}`\n"
    
    if ttl > 0:
        metadata += f"- ⏳ **Calculated TTL**: `{ttl} seconds`\n"
    else:
        metadata += "- ❌ **Action**: Dropped (Score fell below cache_threshold)\n"
    
    # Create an inline element for the metadata so it looks distinct from the text
    metadata_element = cl.Text(name="System Diagnostics", content=metadata, display="inline")
    
    msg.content = output
    msg.elements = [metadata_element]
    
    # Update the UI
    await msg.update()
