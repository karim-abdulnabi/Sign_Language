from imutils.video import VideoStream

camera = VideoStream(src=VIDEO_SOURCE, framerate=FPS).start()

while True:
    frame = camera.read()
    np_array_RGB = opencv2matplotlib(frame)  # Convert to RGB

    image = Image.fromarray(np_array_RGB)  #  PIL image
    byte_array = pil_image_to_byte_array(image)
    client.publish(MQTT_TOPIC_CAMERA, byte_array, qos=MQTT_QOS)
    now = get_now_string()
    print(f"published frame on topic: {MQTT_TOPIC_CAMERA} at {now}")
    time.sleep(1 / FPS)
