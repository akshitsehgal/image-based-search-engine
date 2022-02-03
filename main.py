from deepface import DeepFace
import cv2
import deepface
import numpy as np
import glob
import os
import pickle
from deepface import DeepFace
import streamlit as st
import numpy as np
from PIL import Image
def paginator(label, items, items_per_page=50, on_sidebar=True):
    """Lets the user paginate a set of items.
    Parameters
    ----------
    label : str
        The label to display over the pagination widget.
    items : Iterator[Any]
        The items to display in the paginator.
    items_per_page: int
        The number of items to display per page.
    on_sidebar: bool
        Whether to display the paginator widget on the sidebar.
        
    Returns
    -------
    Iterator[Tuple[int, Any]]
        An iterator over *only the items on that page*, including
        the item's index.
    Example
    -------
    This shows how to display a few pages of fruit.
    >>> fruit_list = [
    ...     'Kiwifruit', 'Honeydew', 'Cherry', 'Honeyberry', 'Pear',
    ...     'Apple', 'Nectarine', 'Soursop', 'Pineapple', 'Satsuma',
    ...     'Fig', 'Huckleberry', 'Coconut', 'Plantain', 'Jujube',
    ...     'Guava', 'Clementine', 'Grape', 'Tayberry', 'Salak',
    ...     'Raspberry', 'Loquat', 'Nance', 'Peach', 'Akee'
    ... ]
    ...
    ... for i, fruit in paginator("Select a fruit page", fruit_list):
    ...     st.write('%s. **%s**' % (i, fruit))
    """

    # Figure out where to display the paginator
    if on_sidebar:
        location = st.sidebar.empty()
    else:
        location = st.empty()

    # Display a pagination selectbox in the specified location.
    items = list(items)
    n_pages = len(items)
    n_pages = (len(items) - 1) // items_per_page + 1
    page_format_func = lambda i: "Page %s" % i
    page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

    # Iterate over the items in the page to let the user display them.
    min_index = page_number * items_per_page
    max_index = min_index + items_per_page
    import itertools
    return itertools.islice(enumerate(items), min_index, max_index)

def load_image(image_file):
	img = Image.open(image_file)
	return img


st.title("Find Similar Images")
st.header("This app finds images similar to the input image")
st.text("Upload an Image")


uploaded_file = st.file_uploader("Choose a food image", type=["jpeg","jpg"])
if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
    st.write(file_details)
    img = load_image(uploaded_file)
    # st.image(img,height=250,width=250)
    with open(os.path.join("image_data",uploaded_file.name),"wb") as f:
      f.write(uploaded_file.getbuffer())
    st.success("Saved File")
    verifications = DeepFace.find('image_data/' + uploaded_file.name,db_path='images/Train',enforce_detection=False)
    paths = verifications['identity'].to_list()
    len_paths = int(0.5 * len(paths))
    paths = paths[:len_paths]
    # image_iterator = paginator("Select a sunset page", paths)
    # indices_on_page, images_on_page = map(list, zip(*image_iterator))
    # st.image(images_on_page, width=200, caption=indices_on_page,use_column_width=200)
    col1,col2,col3,col4 = st.columns(4)
    
    for i in range(0,len_paths-4,4):
        with col1:
            st.image(paths[i])
        with col2:
            st.image(paths[i+1])
        with col3:
            st.image(paths[i+2])
        with col4:
            st.image(paths[i+3])
    
    st.write(verifications)





