# visit http://127.0.0.1:8050/ in your web browser.

import os
import json
import torch
import numpy as np
import plotly.express as px
from PIL import Image
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import torch.nn.functional as F

# Device setup
torch.cuda.empty_cache()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
GRADIENT_PATH = "../gradient-extraction/gradients"
MODEL_PATH = "../models/trained-models/vess_map_custom_cnn.pth"

IMAGE_DIR = "../data/cropped_images"
MAX_IMAGES = None  # Set to an integer to limit the number of images loaded
THRESHOLD_DEFAULT = 0.01
WINDOW_SIZE = 50  # For text display, if needed


# Load the pre-trained model
def load_model(model_path, device):
    """
    Load the pre-trained model.
    """
    import sys

    sys.path.append("../models/")
    from vess_map_custom_cnn import CustomResNet

    model = CustomResNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


MODEL = load_model(MODEL_PATH, DEVICE)


# Load images from directory
def load_images_from_directory(directory_name, device, max_images=None):
    """
    Load images from a directory and return them as a tensor.
    """
    image_files = sorted(os.listdir(directory_name))
    if max_images is not None:
        image_files = image_files[:max_images]

    images = []
    for file_name in image_files:
        if file_name.endswith(".png"):
            img_path = os.path.join(directory_name, file_name)
            img = Image.open(img_path)
            img_array = np.array(img) / 255.0  # Normalize
            images.append(img_array)

    images = np.array(images)
    images_tensor = torch.tensor(images, dtype=torch.float).to(device)
    return images_tensor


ORIGINAL_IMAGES = load_images_from_directory(IMAGE_DIR, DEVICE, MAX_IMAGES)


# Load thresholded gradients
def load_thresholded_gradients(num_images):
    thresholded_gradients = []
    for i in range(num_images):
        grad_path = f"../gradient-extraction/thresholded_gradients/image_{i}.npy"
        if os.path.exists(grad_path):
            grad = np.load(grad_path)
            thresholded_gradients.append(grad)
        else:
            thresholded_gradients.append(np.zeros((128, 128)))
    return np.array(thresholded_gradients)


THRESHOLDED_GRADIENTS = load_thresholded_gradients(len(ORIGINAL_IMAGES))


# Function to plot gradients with bounding box
def plot_gradients_with_bounding_box(gradient, model_name, threshold=0.01):
    """
    Plot the gradient with a bounding box around significant areas.
    """
    gradient = gradient.squeeze()
    mask = np.abs(gradient) > threshold
    non_zero_coords = np.nonzero(mask)

    if len(non_zero_coords[0]) > 0:
        y_min, y_max = non_zero_coords[0].min(), non_zero_coords[0].max()
        x_min, x_max = non_zero_coords[1].min(), non_zero_coords[1].max()
        num_pixels_above_threshold = np.sum(mask)
        bounding_box_area = (y_max - y_min + 1) * (x_max - x_min + 1)
        fulfillment = num_pixels_above_threshold / bounding_box_area

        # Create the figure using Plotly Express
        fig = px.imshow(
            gradient[y_min : y_max + 1, x_min : x_max + 1],
            title=f"Gradient Analysis for {model_name}",
            labels={"x": "x-axis", "y": "y-axis"},
            color_continuous_scale=px.colors.diverging.RdYlGn,
            range_color=[-np.abs(gradient).max(), np.abs(gradient).max()],
        )

        fig.update_layout(
            annotations=[
                {
                    "text": (
                        f"Pixels: {num_pixels_above_threshold}<br>"
                        f"X(min, max): ({x_min}, {x_max})<br>"
                        f"Y(min, max): ({y_min}, {y_max})<br>"
                        f"Area: {bounding_box_area}<br>"
                        f"Fulfillment: {fulfillment:.2f}<br>"
                        f"Threshold: {threshold:.6f}<br>"
                    ),
                    "showarrow": False,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0,
                    "y": 1,
                    "xanchor": "left",
                    "yanchor": "top",
                    "font": {"size": 12, "color": "black"},
                    "bgcolor": "white",
                    "opacity": 0.7,
                }
            ],
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig
    else:
        return px.imshow(
            np.zeros((128, 128)), color_continuous_scale=px.colors.sequential.gray
        )


# Initialize Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(
            dbc.Col(
                html.H1(
                    "Interpreting CNNs - Interactively Gradient Plots",
                    className="text-center",
                ),
                width=12,
            ),
            style={"margin-bottom": "20px"},
        ),
        dcc.Store(id="hover-data-store"),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="image-dropdown",
                        options=[
                            {"label": f"Image {i}", "value": i}
                            for i in range(len(ORIGINAL_IMAGES))
                        ],
                        value=0,
                        style={
                            "text-align": "center",
                            "width": "30vw",
                            "margin-right": "10px",
                            "padding": "5px",  # Add some padding for better appearance
                            "border-radius": "15px",
                        },
                    ),
                    width=1,
                ),
            ],
            style={"margin-bottom": "20px"},
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("Original Image", className="text-center"),
                        dcc.Graph(
                            id="image-display",
                            config={"displayModeBar": True},
                            style={"width": "100%", "height": "70vh"},
                        ),
                    ],
                    width=6,
                    md=6,
                    lg=3,
                ),
                dbc.Col(
                    [
                        html.H3(
                            "Most Significant Pixels",
                            className="text-center",
                        ),
                        dcc.Graph(
                            id="fulfillment-image-display",
                            config={"displayModeBar": True},
                            style={"width": "100%", "height": "70vh"},
                        ),
                    ],
                    width=6,
                    md=6,
                    lg=3,
                ),
                dbc.Col(
                    [
                        html.H3("Model Mask", className="text-center"),
                        dcc.Graph(
                            id="model-mask-display",
                            config={"displayModeBar": True},
                            style={"width": "100%", "height": "70vh"},
                        ),
                    ],
                    width=6,
                    md=6,
                    lg=3,
                ),
                dbc.Col(
                    [
                        html.H3(
                            id="hover-coordinates",
                            children="Coordinates: (x, y)",
                            className="text-center",
                        ),
                        dcc.Graph(
                            id="result-image-display",
                            config={"displayModeBar": True},
                            style={"width": "100%", "height": "70vh"},
                        ),
                    ],
                    width=6,
                    md=6,
                    lg=3,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Input(
                        id="threshold-input",
                        type="number",
                        placeholder="Porcentage of max value of gradient as threshold (default 1%)",
                        min=0.001,
                        max=100,
                        style={
                            "text-align": "center",
                            "width": "30vw",
                            "margin-right": "10px",
                            "padding": "5px",  # Add some padding for better appearance
                            "border-radius": "15px",
                        },
                    ),
                    width=1,
                ),
                dbc.Col(
                    [
                        html.H3(
                            id="threshold-title",
                            children="Detailed Gradient",
                            className="text-center",
                        ),
                        dcc.Graph(
                            id="gradient-display",
                            config={"displayModeBar": True},
                            style={"width": "100%", "height": "70vh"},
                        ),
                    ],
                    width=12,
                    style={"margin-top": "20px"},
                ),
            ]
        ),
    ],
)


# Callback to update the original image based on the dropdown selection
@app.callback(
    Output("image-display", "figure"),
    Output("fulfillment-image-display", "figure"),
    Output("model-mask-display", "figure"),
    Input("image-dropdown", "value"),
)
def update_images(selected_index):
    """
    Update the displayed images based on the selected index.
    """
    original_image_data = ORIGINAL_IMAGES[selected_index].cpu().numpy()

    thresholded_gradients_data = THRESHOLDED_GRADIENTS[selected_index]

    original_fig = px.imshow(
        original_image_data, color_continuous_scale=px.colors.sequential.gray
    )

    with torch.no_grad():
        model_mask = MODEL(ORIGINAL_IMAGES[selected_index].unsqueeze(0).unsqueeze(0))
    softmax_probs = F.softmax(model_mask, dim=1)
    class_one_probs = softmax_probs[0, 1, :, :].cpu().numpy()

    thresholded_gradients_data *= model_mask.argmax(dim=1).squeeze(0).cpu().numpy()
    fulfillment_fig = px.imshow(
        thresholded_gradients_data, color_continuous_scale=px.colors.diverging.RdYlGn
    )

    model_fig = px.imshow(
        class_one_probs, color_continuous_scale=px.colors.diverging.RdYlGn
    )

    return original_fig, fulfillment_fig, model_fig


# Callback to store hover data
@app.callback(
    Output("hover-data-store", "data"),
    Input("image-display", "hoverData"),
    Input("fulfillment-image-display", "hoverData"),
)
def store_hover_data(hover_data_original, hover_data_fulfillment):
    """
    Store hover data from the images.
    """
    hover_data = hover_data_original or hover_data_fulfillment
    if hover_data:
        return json.dumps(
            {"x": hover_data["points"][0]["x"], "y": hover_data["points"][0]["y"]}
        )
    return "{}"


# Callback to update the hover coordinates display
@app.callback(
    Output("hover-coordinates", "children"),
    Input("hover-data-store", "data"),
)
def update_hover_coordinates(hover_data):
    hover_data = json.loads(hover_data)
    if hover_data:
        x, y = hover_data["x"], hover_data["y"]
        return f"Full Gradient: ({x}, {y})"
    return "Full Gradient: (x, y)"


# Callback to update the result image
@app.callback(
    Output("result-image-display", "figure"),
    Input("hover-data-store", "data"),
    Input("image-dropdown", "value"),
)
def update_result_image(hover_data, selected_index):
    """
    Update the result image based on hover data and selected image index.
    """
    hover_data = json.loads(hover_data)
    if hover_data:
        x, y = hover_data["x"], hover_data["y"]
        loaded_gradient = torch.load(
            f"{GRADIENT_PATH}/jacobian_gradient_{selected_index}.pt"
        )

        max_val = torch.max(torch.abs(loaded_gradient)).cpu().item()
        # max_val = np.max([np.max(gradient) for gradient in gradient])
        fig = px.imshow(
            loaded_gradient[y, x].to("cpu"),
            color_continuous_scale=px.colors.diverging.RdYlGn,
            range_color=[-max_val, max_val],
        )
        return fig
    return px.imshow(
        torch.zeros(128, 128), color_continuous_scale=px.colors.diverging.RdYlGn
    )


# Callback to update the gradient display and the threshold title
@app.callback(
    Output("gradient-display", "figure"),
    Output("threshold-title", "children"),
    Input("hover-data-store", "data"),
    Input("threshold-input", "value"),
    Input("image-dropdown", "value"),
)
def update_gradient_display(hover_data_json, threshold, selected_index):
    """
    Update the gradient display based on hover data and threshold.
    """
    hover_data = json.loads(hover_data_json)
    if hover_data:
        x, y = hover_data["x"], hover_data["y"]
        loaded_gradient = torch.load(
            f"{GRADIENT_PATH}/jacobian_gradient_{selected_index}.pt"
        )
        max_val = torch.max(torch.abs(loaded_gradient)).cpu().item()

        threshold = threshold if threshold is not None else 0.01
        threshold_val = (threshold / 100) * max_val

        # Return the gradient with the diverging color map
        fig = plot_gradients_with_bounding_box(
            loaded_gradient[x, y].to("cpu").numpy(), "Model", threshold=threshold_val
        )
        fig.update_layout(title=f"Pixels of gradient above: {threshold_val:.4f}")
        return fig, [
            f"Detailed Thresholded Gradient",
            html.Br(),
            f"Threshold = {threshold_val:.4f}",
        ]
    return (
        px.imshow(
            torch.zeros(128, 128), color_continuous_scale=px.colors.diverging.RdYlGn
        ),
        "Detailed Gradient:",
    )


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
