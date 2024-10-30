import cv2
import numpy as np
import matplotlib.path as mpltPath


def is_point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon.

    :param point: A tuple representing the point (x, y).
    :param polygon: A list of tuples representing the polygon vertices [(x1, y1), (x2, y2), ...].
    :return: True if the point is inside the polygon, False otherwise.
    """
    # Convert to numpy array for compatibility with mpltPath
    polygon = np.array(polygon)

    # Create a Path object for the polygon
    path = mpltPath.Path(polygon)

    # Check if the point is inside the polygon
    return path.contains_point(point)


def show_point_and_polygon_cv2(point, polygon, result, canvas_size=(500, 500)):
    """
    Visualize the point and the polygon using OpenCV.

    :param point: A tuple representing the point (x, y).
    :param polygon: A list of tuples representing the polygon vertices [(x1, y1), (x2, y2), ...].
    :param result: Boolean indicating if the point is inside the polygon.
    :param canvas_size: Size of the canvas (width, height).
    """
    # Create a white canvas
    canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255

    # Convert polygon to integer coordinates
    polygon_pts = np.array(polygon, dtype=np.int32)

    # Draw the polygon
    cv2.polylines(canvas, [polygon_pts], isClosed=True, color=(0, 0, 0), thickness=2)

    # Draw the point
    color = (0, 255, 0) if result else (0, 0, 255)  # Green if inside, red if outside
    cv2.circle(canvas, (int(point[0]), int(point[1])), radius=5, color=color, thickness=-1)

    # Add text
    label = "Inside" if result else "Outside"
    cv2.putText(canvas, label, (int(point[0]) + 10, int(point[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Show the image
    cv2.imshow("Point in Polygon Check", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


