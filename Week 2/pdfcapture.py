import cv2
import numpy as np
import subprocess  # For macOS screencapture
import time
import os
import pyautogui
from fpdf import FPDF  # for saving images as PDF
from PIL import Image
import math
import threading
from pynput import keyboard
import pygetwindow as gw
import pytesseract

def is_retina():
    """Check if the system is using a Retina display."""
    try:
        result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        if 'Retina' in output:
            print("Retina display detected.")
            return True
    except Exception as e:
        print(f"Error checking for Retina display: {e}")
    return False

def capture_screenshot(region, output_file="screenshot.png"):
    """Capture the entire screen and crop to the specified region."""
    try:
        x, y, width, height = region

        # Capture the entire screen using screencapture
        print("Capturing fullscreen screenshot...")
        full_screenshot_path = "fullscreen.png"
        subprocess.run(["screencapture", full_screenshot_path])

        # Load the fullscreen image using OpenCV
        full_image = cv2.imread(full_screenshot_path)
        if full_image is None:
            raise Exception(f"Could not load fullscreen screenshot: {full_screenshot_path}")

        # Crop the image to the selected region
        cropped_image = full_image[y:y+height, x:x+width]

        # Save the cropped image
        cv2.imwrite(output_file, cropped_image)
        print(f"Cropped screenshot saved as {output_file} with shape: {cropped_image.shape}")
        return cropped_image
    except Exception as e:
        print(f"Error capturing and cropping screenshot: {e}")
        return None

def select_capture_region():
    """Allow the user to select a region of the screen using a bounding box."""
    print("Select the region of the screen you want to capture.")
    time.sleep(2)  # Give the user time to focus on the window to capture

    screenshot = pyautogui.screenshot()
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Use OpenCV to allow the user to draw a bounding box (Resizable ROI)
    region = cv2.selectROI("Select Region", screenshot_cv, showCrosshair=True)
    cv2.destroyWindow("Select Region")

    if all(region):  # If a valid region is selected
        print(f"Region selected: {region}")
        return region
    else:
        print("No region selected. Exiting...")
        exit()

def crop_overlap(prev_image, current_image):
    """Identify and crop overlapping areas between previous and current images."""
    try:
        # Define the region of the previous image to be used as a template for matching
        overlap_height = int(prev_image.shape[0] * 0.15)  # Use bottom 15% of the previous image
        template = prev_image[-overlap_height:, :]  # Bottom portion of the previous image

        # Perform template matching to find the overlap region
        result = cv2.matchTemplate(current_image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Define a threshold to consider the match reliable
        threshold = 0.8  # You may need to tweak this depending on how the images look

        if max_val > threshold:
            # Crop the current image from the point where the overlap ends
            y_match = max_loc[1] + overlap_height  # Overlap ends at this y-coordinate
            cropped_image = current_image[y_match:, :]  # Crop from where overlap ends
            print(f"Overlap found. Cropping current image from y = {y_match}.")
        else:
            # If no significant overlap is found, return the original current image
            print("No significant overlap found. Returning the original current image.")
            cropped_image = current_image

        return cropped_image

    except Exception as e:
        print(f"Error in crop_overlap: {e}")
        return current_image

def scroll_capture(region):
    frame_count = 0
    prev_image = None
    stitched_image = None
    captured_images = []

    while True:
        try:
            # Capture screenshot of the selected region
            current_image = capture_screenshot(region, output_file=f"screenshot_{frame_count}.png")

            if current_image is None or current_image.size == 0:
                print(f"Failed to capture image at frame {frame_count}.")
                continue  # Skip if image wasn't captured

            print(f"Frame {frame_count}: Captured image shape: {current_image.shape}")

            if prev_image is not None:
                non_overlap_image = crop_overlap(prev_image, current_image)

                if stitched_image is None:
                    stitched_image = prev_image  # Start with the first image
                    print(f"Starting stitching with shape: {stitched_image.shape}")

                # Append the new image to the bottom of the stitched image
                try:
                    stitched_image = np.vstack((stitched_image, non_overlap_image))
                    print(f"Stitched image updated. Current stitched image shape: {stitched_image.shape}")
                except Exception as e:
                    print(f"Error during stitching: {e}")
            else:
                stitched_image = current_image

            prev_image = current_image
            captured_images.append(f"screenshot_{frame_count}.png")
            frame_count += 1

            # Display the current captured image
            window_name = "Current Capture"
            cv2.imshow(window_name, current_image)

            # Minimize the window after displaying it
            time.sleep(0.5)  # Short delay to ensure the window is created
            try:
                # Get a list of all window titles
                window_titles = gw.getAllTitles()

                # Find and minimize the OpenCV window by title
                for title in window_titles:
                    if window_name in title:
                        window = gw.getWindowsWithTitle(title)[0]
                        window.minimize()
                        print(f"Window '{window_name}' minimized.")
                        break
            except IndexError:
                print(f"Could not find window '{window_name}' to minimize.")
            except Exception as e:
                print(f"Error minimizing window: {e}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit on 'q'

            # Wait for 0.1 seconds before capturing the next frame
            #time.sleep(0.1)

        except KeyboardInterrupt:
            print("Process interrupted. Saving stitched image and PDF...")
            break

    if stitched_image is None or stitched_image.size == 0:
        print("No images were captured or stitched.")
        return

    # Save the final stitched image
    try:
        cv2.imwrite("final_stitched_image.png", stitched_image)
        print("Stitched image saved as 'final_stitched_image.png'.")
    except Exception as e:
        print(f"Error saving stitched image: {e}")

def pre_process_image(image):
    """
    Pre-process the image to enhance OCR detection by improving contrast, reducing noise, and performing edge detection.
    """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use edge detection to highlight contours (important for formula and text clarity)
    edges = cv2.Canny(blurred, 100, 200)
    # Return the processed image to be used in OCR
    return edges

def check_and_adjust_content(cropped_image, page_height_px):
    """
    Detect if any text blocks, formulas, or graphs are split between pages and adjust them to prevent splitting.
    Uses OCR for text, formulas, and image processing for graphs to ensure no content is split between pages.
    """
    try:
        # Pre-process the image before running OCR
        processed_image = pre_process_image(cropped_image)

        # Run OCR on the processed image to detect text and their positions
        ocr_data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)

        # Variables to detect block regions for text/formulas
        block_top = None
        block_bottom = None
        block_texts = []
        formula_detected = False
        previous_bottom = 0

        # Minimum vertical gap between lines to treat them as part of the same block
        line_spacing_threshold = 15

        # Symbols that might be part of a formula
        formula_keywords = ['∫', '=', '∑', 'π', 'dx', 'dy', 'dz', 'sin', 'cos', 'tan', 'log', 'ln', 'exp', '^', '/', '√']

        # Iterate through all detected text lines
        for i in range(len(ocr_data['text'])):
            current_text = ocr_data['text'][i].strip()

            if current_text:  # Ignore empty lines
                top = ocr_data['top'][i]
                height = ocr_data['height'][i]
                bottom = top + height

                # Check if this line contains a formula-like expression
                if any(keyword in current_text for keyword in formula_keywords):
                    formula_detected = True  # Treat this as part of a formula block

                # If the gap between the current line and the previous line is small, treat them as part of the same block
                if block_top is None or (top - previous_bottom) < line_spacing_threshold:
                    if block_top is None:
                        block_top = top
                    block_bottom = bottom
                    block_texts.append(current_text)
                else:
                    # We've encountered a new block, so we check if the previous block is near the page break
                    # Use larger margin for formulas to ensure no splitting
                    margin = 120 if formula_detected else 100
                    if block_bottom > page_height_px - margin:
                        print(f"Detected block spanning the page break: {block_texts}. Moving entire block.")
                        return True  # Move entire block to the next page
                    
                    # Reset for the new block
                    block_top = top
                    block_bottom = bottom
                    block_texts = [current_text]
                    formula_detected = False  # Reset formula detection

                previous_bottom = bottom

        # Now handle non-text content like graphs using basic image processing
        gray_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2GRAY)

        # Detect non-white areas (where visual content like graphs exist)
        _, binary_image = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)

        # Find contours of the non-white regions
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any large contour (like a graph) is near the bottom of the page
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # If the graph's height is greater than the remaining space on the page, move it
            remaining_space = page_height_px - y

            # Adjust for graphs varying in size
            if h > remaining_space:
                print(f"Detected large graph (height: {h}px) that spans the page break. Moving to the next page.")
                return True  # Move the graph to the next page
            else:
                print(f"Graph detected (height: {h}px) fits within the remaining space of {remaining_space}px.")

        return False  # No split detected
    except Exception as e:
        print(f"Error during content detection: {e}")
        return False

def save_stitched_image_as_pdf(stitched_image_path, output_pdf="partitioned_output.pdf", page_height_mm=297, dpi=96):
    """Save the stitched image as a multi-page PDF, partitioning it based on page height and adding white space on the last page."""
    try:
        img = Image.open(stitched_image_path)
        width_px, height_px = img.size

        # Convert pixel size to millimeters based on dpi
        width_mm = (width_px / dpi) * 25.4
        page_height_px = int((page_height_mm / 25.4) * dpi)

        num_pages = math.ceil(height_px / page_height_px)
        pdf = FPDF(unit="mm", format=[width_mm, page_height_mm])

        for page in range(num_pages - 1):  # Process all pages except the last one
            upper = page * page_height_px
            lower = min((page + 1) * page_height_px, height_px)
            box = (0, upper, width_px, lower)

            cropped_img = img.crop(box)

            # Check and adjust content before rendering the block to the PDF
            adjust_content = check_and_adjust_content(cropped_img, page_height_px)
            
            if adjust_content:
                print(f"Adjusting page content at page {page}.")
                # You can skip this page or handle the logic to move content to the next one

            temp_img_path = f"temp_page_{page}.png"
            cropped_img.save(temp_img_path)

            pdf.add_page()
            pdf.image(temp_img_path, 0, 0, width_mm, page_height_mm)

        # Handle the last page separately with white space if needed
        last_page_height = height_px % page_height_px
        if last_page_height > 0:
            # Add white space to fill the remaining area
            white_space_height_px = page_height_px - last_page_height
            white_space = Image.new('RGB', (width_px, white_space_height_px), (255, 255, 255))  # Create a white image

            # Crop the last part of the image
            upper = (num_pages - 1) * page_height_px
            cropped_img = img.crop((0, upper, width_px, height_px))

            # Combine the last image with the white space
            final_img = Image.new('RGB', (width_px, page_height_px), (255, 255, 255))
            final_img.paste(cropped_img, (0, 0))  # Paste the cropped content at the top
            final_img.paste(white_space, (0, last_page_height))  # Add the white space below

            # Save the new image with white space as the final page
            temp_img_path = f"temp_page_{num_pages}.png"
            final_img.save(temp_img_path)

            pdf.add_page()
            pdf.image(temp_img_path, 0, 0, width_mm, page_height_mm)
        else:
            # If the last page doesn't need white space, just add it normally
            upper = (num_pages - 1) * page_height_px
            lower = height_px
            box = (0, upper, width_px, lower)

            cropped_img = img.crop(box)
            temp_img_path = f"temp_page_{num_pages}.png"
            cropped_img.save(temp_img_path)

            pdf.add_page()
            pdf.image(temp_img_path, 0, 0, width_mm, page_height_mm)

        # Output the final PDF
        pdf_output_path = output_pdf
        pdf.output(pdf_output_path)
        print(f"PDF saved as '{pdf_output_path}'.")

    except Exception as e:
        print(f"Error partitioning PNG to PDF: {e}")


if __name__ == "__main__":
    # Step 1: Select the region to capture
    time.sleep(3)
    region = select_capture_region()

    # Step 2: Check if we are on a Retina display (though not needed for scaling anymore)
    if is_retina():
        print("Retina display detected.")
    
    time.sleep(3)
    # Step 3: Start scrolling and capturing within the selected region
    try:
        scroll_capture(region)  # Removed scale_factor argument
    except KeyboardInterrupt:
        print("Process interrupted. Saving stitched image...")

    # Save the stitched image as a single-page PDF
    save_stitched_image_as_pdf("final_stitched_image.png")

    # Close the OpenCV window after all captures are done
    cv2.destroyAllWindows()