import gradio as gr
import requests
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_ENDPOINTS = {
    "chat": f"{API_BASE_URL}/api/v1/chat",
    "query": f"{API_BASE_URL}/api/v1/query",
    "generate": f"{API_BASE_URL}/api/v1/generate-cvs",
    "stats": f"{API_BASE_URL}/api/v1/stats",
    "templates": f"{API_BASE_URL}/api/v1/templates/random"
}

def check_backend_health() -> Dict[str, Any]:
    """Check if the backend is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/stats", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "data": response.json()}
        else:
            return {"status": "error", "message": f"Backend returned status {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Cannot connect to backend. Is it running?"}
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "Backend connection timeout"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

def chat_fn(message: str, history: List[List[str]]) -> List[List[str]]:
    """Handle chat messages with the backend."""
    if not message.strip():
        return history
    
    try:
        response = requests.post(
            API_ENDPOINTS["chat"], 
            json={"message": message, "session_id": "gradio_user"},
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        if result.get("success"):
            reply = result.get("reply", "No reply received")
            # Add search stats if available
            stats = result.get("search_stats", {})
            if stats:
                reply += f"\n\n📊 Search: {stats.get('total_results', 0)} results in {stats.get('search_time', 0):.3f}s"
            # Convert to messages format
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})
        else:
            error_msg = f"⚠️ Backend error: {result.get('detail', 'Unknown error')}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            
    except requests.exceptions.ConnectionError:
        error_msg = "⚠️ Cannot connect to backend. Please ensure the backend is running on port 8000."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
    except requests.exceptions.Timeout:
        error_msg = "⚠️ Request timeout. The backend is taking too long to respond."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
    except requests.exceptions.RequestException as e:
        error_msg = f"⚠️ Network error: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
    except Exception as e:
        error_msg = f"⚠️ Unexpected error: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
    
    return history

def query_cvs_fn(query: str, num_results: int = 3) -> str:
    """Query CVs directly without chat."""
    if not query.strip():
        return "Please enter a search query."
    
    try:
        response = requests.post(
            API_ENDPOINTS["query"],
            json={"question": query},
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        if result.get("success"):
            results = result.get("results", [])
            if not results:
                return "No CVs found matching your query."
            
            output = f"🔍 Found {len(results)} CV(s) for: '{query}'\n\n"
            for i, cv in enumerate(results, 1):
                content = cv.get("content", "")[:200] + "..." if len(cv.get("content", "")) > 200 else cv.get("content", "")
                score = cv.get("score", 0)
                output += f"**{i}. Score: {score:.3f}**\n{content}\n\n"
            
            return output
        else:
            return f"⚠️ Backend error: {result.get('detail', 'Unknown error')}"
            
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def generate_cvs_fn(count: int) -> str:
    """Generate new CVs."""
    if count <= 0 or count > 10:
        return "Please enter a number between 1 and 10."
    
    try:
        response = requests.post(
            API_ENDPOINTS["generate"],
            json={"count": count},
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        if result.get("success"):
            cvs = result.get("cvs", [])
            output = f"✅ Generated {len(cvs)} CV(s)\n\n"
            for i, cv in enumerate(cvs, 1):
                template = cv.get("template", {})
                output += f"**{i}. {template.get('role', 'Unknown')} - {template.get('level', 'Unknown')}**\n"
                output += f"   File: {cv.get('filename', 'Unknown')}\n"
                if cv.get('image_validation'):
                    validation = cv['image_validation']
                    if validation.get('is_valid'):
                        output += f"   ✅ Image validation passed\n"
                    else:
                        output += f"   ⚠️ Image validation issues\n"
                output += "\n"
            return output
        else:
            return f"⚠️ Backend error: {result.get('detail', 'Unknown error')}"
            
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def get_stats_fn() -> str:
    """Get system statistics."""
    try:
        # Get main stats
        response = requests.get(API_ENDPOINTS["stats"], timeout=10)
        response.raise_for_status()
        
        result = response.json()
        if not result.get("success"):
            return f"⚠️ Backend error: {result.get('detail', 'Unknown error')}"
        
        # Get template count
        template_count = 0
        try:
            template_response = requests.get(API_ENDPOINTS["templates"], timeout=10)
            if template_response.status_code == 200:
                template_result = template_response.json()
                if template_result.get("success"):
                    template_count = len(template_result.get("templates", []))
        except:
            pass  # If template endpoint fails, just show 0
        
        # Parse the actual API response structure
        system = result.get("system", {})
        index = result.get("index", {})
        rag = result.get("rag", {})
        
        output = "📊 **System Statistics**\n\n"
        output += "**📂 CV Database:**\n"
        output += f"• Total CVs: {index.get('processed_files', 0)}\n"
        output += f"• Documents indexed: {index.get('total_documents', 0)}\n"
        output += f"• Index size: {index.get('index_size', 0)} vectors\n\n"
        
        output += "**🎯 Templates:**\n"
        output += f"• Available templates: {template_count}\n\n"
        
        output += "**🔍 Search Engine:**\n"
        output += f"• Vector dimensions: {index.get('dimensions', 0)}\n"
        output += f"• Cache enabled: {'Yes' if rag.get('cache_enabled') else 'No'}\n"
        output += f"• Cache size: {rag.get('cache_size', 0)}\n"
        output += f"• Max search results: {rag.get('max_k', 0)}\n\n"
        
        output += "**⚙️ System:**\n"
        output += f"• Environment: {system.get('environment', 'Unknown')}\n"
        output += f"• Debug mode: {'Yes' if system.get('debug') else 'No'}\n"
        output += f"• Version: {system.get('version', 'Unknown')}\n"
        
        return output
            
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def create_interface():
    """Create the Gradio interface."""
    # Check backend health
    health = check_backend_health()
    
    with gr.Blocks(
        title="AI-Powered CV Engine",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    ) as demo:
        gr.Markdown("# 🤖 AI-Powered CV Engine")
        
        # Status indicator
        if health["status"] == "healthy":
            gr.Markdown("🟢 **Backend Status: Connected**")
        else:
            gr.Markdown(f"🔴 **Backend Status: {health.get('message', 'Unknown')}**")
        
        with gr.Tabs():
            # Chat Tab
            with gr.TabItem("💬 Chat with CVs"):
                gr.Markdown("Ask questions about the CVs in the database.")
                chatbot = gr.Chatbot(height=400, type="messages")
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="e.g., 'Who has Python experience?' or 'Find senior developers'"
                )
                clear = gr.Button("Clear")
                
                msg.submit(chat_fn, [msg, chatbot], [chatbot])
                clear.click(lambda: [], None, chatbot, queue=False)
            
            # Search Tab
            with gr.TabItem("🔍 Search CVs"):
                gr.Markdown("Search for specific CVs without chat.")
                search_query = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g., 'software engineer with React experience'"
                )
                num_results = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1,
                    label="Number of Results"
                )
                search_btn = gr.Button("Search")
                search_output = gr.Textbox(label="Results", lines=10)
                
                search_btn.click(query_cvs_fn, [search_query, num_results], search_output)
            
            # Generate Tab
            with gr.TabItem("⚡ Generate CVs"):
                gr.Markdown("Generate new CVs with AI.")
                cv_count = gr.Slider(
                    minimum=1, maximum=10, value=1, step=1,
                    label="Number of CVs to Generate"
                )
                generate_btn = gr.Button("Generate CVs")
                generate_output = gr.Textbox(label="Generation Results", lines=10)
                
                generate_btn.click(generate_cvs_fn, [cv_count], generate_output)
            
            # Stats Tab
            with gr.TabItem("📊 Statistics"):
                gr.Markdown("View system statistics and health.")
                stats_btn = gr.Button("Refresh Statistics")
                stats_output = gr.Textbox(label="System Stats", lines=8)
                
                stats_btn.click(get_stats_fn, [], stats_output)
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("**CV Engine** - AI-powered CV generation and analysis system")
    
    return demo

def launch_ui():
    """Launch the Gradio interface."""
    demo = create_interface()
    demo.launch(
        server_port=7860,
        server_name="0.0.0.0",
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    launch_ui()