from io import BytesIO
import base64
from PIL import Image
import numpy as np

def get_image_download_link(img):
    """Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
    img = Image.fromarray((img * 255).astype(np.uint8))
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}">Download Image</a>'
    return href