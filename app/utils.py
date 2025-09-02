from fastapi import UploadFile
import aiofiles
import os

async def save_upload_file(upload_file: UploadFile, destination: str):
    """Save uploaded file to destination"""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    async with aiofiles.open(destination, 'wb') as f:
        content = await upload_file.read()
        await f.write(content)

def create_context_from_results(search_results):
    """Create context string from vector search results"""
    context = ""
    for i, match in enumerate(search_results.matches):
        metadata = match.metadata
        context += f"""
        Case {i+1}:
        Disease: {metadata.get('disease', 'Unknown')}
        Description: {metadata.get('description', 'No description')}
        Confidence: {match.score:.2f}
        ---
        """
    return context