package com.depthsightai;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.media.Image;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.RadioGroup;

import com.depthsightai.R;
import com.google.android.material.chip.Chip;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Camera;
import com.google.ar.core.Config;
import com.google.ar.core.Frame;
import com.google.ar.core.Session;
import com.google.ar.core.TrackingState;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.UnavailableException;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class MainActivity extends Activity implements GLSurfaceView.Renderer {
    private static final String TAG = "MainActivity";
    private static final int REQUEST_CAMERA_PERMISSION = 100;

    private NanoDetNcnn nanodetncnn = new NanoDetNcnn();

    private Session session;
    private GLSurfaceView surfaceView;
    private SurfaceView overlayView;
    private TextView resultTextView;
    private TextView fpsTextView;
    private Chip detectionCountChip;
    private boolean installRequested;
    private boolean depthEnabled = true;

    private int current_model = 0;
    private int current_cpugpu = 0;

    private int textureId = -1;

    // Background Renderer
    private FloatBuffer quadVertices;
    private FloatBuffer quadTexCoord;
    private FloatBuffer quadTexCoordTransformed;
    private int quadProgram;
    private int quadPositionParam;
    private int quadTexCoordParam;
    private int quadTextureParam;

    // Concurrency control for inference
    private boolean isComputing = false;
    private NanoDetNcnn.Box[] latestBoxes = null;

    private float screenWidth;
    private float screenHeight;

    private static final String[] class_names = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        setTheme(R.style.AppTheme); // switch out of SplashTheme
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        surfaceView = findViewById(R.id.cameraview_gl);
        surfaceView.setPreserveEGLContextOnPause(true);
        surfaceView.setEGLContextClientVersion(2);
        surfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0);
        surfaceView.setRenderer(this);
        surfaceView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);

        overlayView = findViewById(R.id.cameraview_overlay);
        overlayView.setZOrderOnTop(true);
        overlayView.setZOrderOnTop(true);
        overlayView.getHolder().setFormat(PixelFormat.TRANSPARENT);

        resultTextView = findViewById(R.id.result_textView);
        fpsTextView = findViewById(R.id.fps_text);
        detectionCountChip = findViewById(R.id.detection_count_chip);

        Spinner spinnerModel = findViewById(R.id.spinnerModel);
        if (spinnerModel != null) {
            spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
                @Override
                public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
                    if (position != current_model) {
                        current_model = position;
                        reload();
                    }
                }

                @Override
                public void onNothingSelected(AdapterView<?> arg0) {
                }
            });
        }

        Spinner spinnerCPUGPU = findViewById(R.id.spinnerCPUGPU);
        if (spinnerCPUGPU != null) {
            spinnerCPUGPU.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
                @Override
                public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
                    if (position != current_cpugpu) {
                        current_cpugpu = position;
                        reload();
                    }
                }

                @Override
                public void onNothingSelected(AdapterView<?> arg0) {
                }
            });
        }

        RadioGroup radioGroupMode = findViewById(R.id.radioGroupMode);
        if (radioGroupMode != null) {
            radioGroupMode.setOnCheckedChangeListener((group, checkedId) -> {
                depthEnabled = (checkedId == R.id.radioObjectDepth);
            });
        }

        installRequested = false;
        reload();

        setupGlQuad();
    }

    private void reload() {
        boolean ret_init = nanodetncnn.loadModel(getAssets(), current_model, current_cpugpu);
        if (!ret_init) {
            Log.e(TAG, "nanodetncnn loadModel failed");
        }
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (session == null) {
            Exception exception = null;
            String message = null;
            try {
                switch (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                    case INSTALL_REQUESTED:
                        installRequested = true;
                        return;
                    case INSTALLED:
                        break;
                }

                if (ContextCompat.checkSelfPermission(this,
                        Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
                    ActivityCompat.requestPermissions(this, new String[] { Manifest.permission.CAMERA },
                            REQUEST_CAMERA_PERMISSION);
                    return;
                }

                session = new Session(this);
                Config config = new Config(session);
                config.setDepthMode(Config.DepthMode.AUTOMATIC);
                session.configure(config);

            } catch (UnavailableException e) {
                message = "ARCore is unavailable: " + e.getMessage();
                exception = e;
            } catch (Exception e) {
                message = "Failed to create AR session: " + e.getMessage();
                exception = e;
            }

            if (message != null) {
                Toast.makeText(this, message, Toast.LENGTH_LONG).show();
                Log.e(TAG, "Exception creating session", exception);
                return;
            }
        }

        try {
            session.resume();
        } catch (CameraNotAvailableException e) {
            Toast.makeText(this, "Camera not available. Try restarting the app.", Toast.LENGTH_LONG).show();
            session = null;
            return;
        }

        if (screenWidth > 0 && screenHeight > 0) {
            session.setDisplayGeometry(getWindowManager().getDefaultDisplay().getRotation(), (int) screenWidth,
                    (int) screenHeight);
        }

        if (surfaceView != null) {
            surfaceView.onResume();
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        if (session != null) {
            session.pause();
        }
        if (surfaceView != null) {
            surfaceView.onPause();
        }
    }

    @Override
    protected void onDestroy() {
        if (session != null) {
            session.close();
            session = null;
        }
        super.onDestroy();
    }

    // --- GLSurfaceView.Renderer ---

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        textureId = textures[0];
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);

        if (session != null) {
            session.setCameraTextureName(textureId);
        }

        compileGlProgram();
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        GLES20.glViewport(0, 0, width, height);
        if (session != null) {
            session.setDisplayGeometry(getWindowManager().getDefaultDisplay().getRotation(), width, height);
        }
        screenWidth = width;
        screenHeight = height;
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);

        if (session == null)
            return;

        session.setCameraTextureName(textureId);

        Frame frame;
        try {
            frame = session.update();
        } catch (CameraNotAvailableException e) {
            Log.e(TAG, "Camera not available during onDrawFrame", e);
            return;
        }

        Camera camera = frame.getCamera();

        if (frame.hasDisplayGeometryChanged()) {
            quadTexCoord.position(0);
            quadTexCoordTransformed.position(0);
            frame.transformDisplayUvCoords(quadTexCoord, quadTexCoordTransformed);
        }

        if (camera.getTrackingState() != TrackingState.TRACKING) {
            drawBackground();
            drawOverlay(); // Keep drawing UI text/boxes to avoid starvation
            return;
        }

        // 1. Draw camera background
        drawBackground();

        // 2. Inference: Process NanoDet if free
        if (!isComputing) {
            boolean didStartThread = false;
            Image cameraImage = null;
            Image depthImage = null;
            try {
                cameraImage = frame.acquireCameraImage();
                try {
                    depthImage = frame.acquireDepthImage16Bits();
                } catch (Exception e) {
                    // Depth image not available yet, ignore and continue without depth
                }

                if (cameraImage != null) {
                    isComputing = true;
                    didStartThread = true;
                    final Image fCameraImage = cameraImage;
                    final Image fDepthImage = depthImage;

                    // Run inference in background to prevent GL thread stutter
                    new Thread(() -> {
                        try {
                            processFrame(fCameraImage, fDepthImage);
                        } catch (Exception e) {
                            Log.e(TAG, "Error in processFrame", e);
                        } finally {
                            fCameraImage.close();
                            if (fDepthImage != null)
                                fDepthImage.close();
                            isComputing = false;
                        }
                    }).start();
                }
            } catch (Exception e) {
                // If acquireCameraImage fails
                if (cameraImage != null) {
                    cameraImage.close();
                }
            }
        }

        // 3. Draw Overlays
        drawOverlay();
    }

    private void processFrame(Image cameraImage, Image depthImage) {
        int width = cameraImage.getWidth();
        int height = cameraImage.getHeight();

        Image.Plane[] planes = cameraImage.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] yBytes = new byte[ySize];
        byte[] uBytes = new byte[uSize];
        byte[] vBytes = new byte[vSize];

        yBuffer.get(yBytes);
        uBuffer.get(uBytes);
        vBuffer.get(vBytes);

        int yRowStride = planes[0].getRowStride();
        int uvRowStride = planes[1].getRowStride();
        int uvPixelStride = planes[1].getPixelStride();

        // Screen rotation could be passed properly, assuming 90 (portrait)
        int displayRotation = getWindowManager().getDefaultDisplay().getRotation();
        int cameraRotation = 90; // Typical rear ARCore camera portrait

        // Run detection
        NanoDetNcnn.Box[] boxes = nanodetncnn.detectYuv(
                yBytes, uBytes, vBytes, yRowStride, uvRowStride, uvPixelStride, width, height, cameraRotation);

        if (boxes != null && depthImage != null && depthEnabled) {
            int depthW = depthImage.getWidth();
            int depthH = depthImage.getHeight();
            Image.Plane depthPlane = depthImage.getPlanes()[0];
            ByteBuffer depthBuffer = depthPlane.getBuffer();
            int depthRowStride = depthPlane.getRowStride();

            // Resolution scaling logic between RGB inference image and Depth Image
            // We rotated the image to portrait in C++, so the box coordinates are in
            // portrait
            // Let's assume portrait
            int inferW = (cameraRotation == 90 || cameraRotation == 270) ? height : width;
            int inferH = (cameraRotation == 90 || cameraRotation == 270) ? width : height;

            // Compute depth for each box
            for (NanoDetNcnn.Box box : boxes) {
                // Map box to depth image coordinates.
                // Note: ARCore's depthImage is oriented the same as display if display geometry
                // is set.
                // It is usually smaller (e.g. 160x120 or 640x480).

                float scaleX = (float) depthW / inferW;
                float scaleY = (float) depthH / inferH;

                int dx0 = (int) (box.x0 * scaleX);
                int dy0 = (int) (box.y0 * scaleY);
                int dx1 = (int) (box.x1 * scaleX);
                int dy1 = (int) (box.y1 * scaleY);

                dx0 = Math.max(0, dx0);
                dy0 = Math.max(0, dy0);
                dx1 = Math.min(depthW - 1, dx1);
                dy1 = Math.min(depthH - 1, dy1);

                float depthMeters = computeMedianDepth(depthBuffer, depthRowStride, dx0, dy0, dx1, dy1);
                box.prob = depthMeters; // Storing depth in meters temporarily in prob!
            }
        }

        latestBoxes = boxes;
    }

    private float computeMedianDepth(ByteBuffer depthBuffer, int rowStride, int x0, int y0, int x1, int y1) {
        int width = x1 - x0;
        int height = y1 - y0;
        if (width <= 0 || height <= 0)
            return 0f;

        short[] samples = new short[100];
        int sampleCount = 0;

        int stepY = Math.max(1, height / 10);
        int stepX = Math.max(1, width / 10);

        depthBuffer.order(ByteOrder.nativeOrder()); // Ensure proper byte order before reading raw shorts

        for (int y = y0; y < y1 && sampleCount < 100; y += stepY) {
            for (int x = x0; x < x1 && sampleCount < 100; x += stepX) {
                int index = (y * rowStride) + (x * 2);

                if (index >= 0 && index + 1 < depthBuffer.limit()) {
                    short depthSample = depthBuffer.getShort(index);

                    // valid depth is up to 8 meters usually. (ARCore drops bits above 13 for
                    // confidence sometimes, depending on DepthMode API)
                    // The lower 13 bits are the depth value in mm.
                    short depthMm = (short) (depthSample & 0x1FFF);

                    if (depthMm != 0) {
                        samples[sampleCount++] = depthMm;
                    }
                }
            }
        }

        if (sampleCount == 0)
            return 0f;

        Arrays.sort(samples, 0, sampleCount);
        short medianDepth = samples[sampleCount / 2];

        // 16-bit depth is in millimeters
        return medianDepth / 1000.0f;
    }

    private void drawOverlay() {
        if (overlayView == null)
            return;

        SurfaceHolder holder = overlayView.getHolder();
        Canvas canvas = holder.lockCanvas();
        if (canvas == null)
            return;

        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);

        if (latestBoxes != null && screenWidth > 0 && screenHeight > 0) {
            Paint paint = new Paint();
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(5f);

            Paint textPaint = new Paint();
            textPaint.setTextSize(60f);
            textPaint.setColor(Color.WHITE);
            textPaint.setStyle(Paint.Style.FILL);

            Paint bgPaint = new Paint();
            bgPaint.setStyle(Paint.Style.FILL);

            // Assume inference size is similar to screen aspect ratio and scaled up
            // Note: The inference coords need scale mapping to the screen coords!
            // We assume 1080x1920 or similar.
            // float cameraW = 1080; // approximate
            // float cameraH = 1920;

            // Hardcode scale for demo since we know default ncnn is 640x480 but flipped ->
            // 480x640
            float scaleX = screenWidth / 480.0f;
            float scaleY = screenHeight / 640.0f;

            StringBuilder sb = new StringBuilder();

            for (NanoDetNcnn.Box box : latestBoxes) {
                float left = box.x0 * scaleX;
                float top = box.y0 * scaleY;
                float right = box.x1 * scaleX;
                float bottom = box.y1 * scaleY;

                paint.setColor(Color.GREEN);
                bgPaint.setColor(Color.argb(150, 0, 0, 0));

                canvas.drawRect(left, top, right, bottom, paint);

                String label = box.label >= 0 && box.label < class_names.length ? class_names[box.label] : "unknown";
                float val = box.prob;

                String text;
                if (depthEnabled) {
                    text = String.format("%s %.2f m", label, val);
                } else {
                    text = String.format("%s %d%%", label, (int) (val * 100));
                }

                canvas.drawRect(left, top - 80, left + textPaint.measureText(text) + 20, top, bgPaint);
                canvas.drawText(text, left + 10, top - 20, textPaint);

                if (depthEnabled) {
                    sb.append("• ").append(label).append(" (").append(String.format("%.2f", val)).append("m)\n");
                } else {
                    sb.append("• ").append(label).append(" (").append((int) (val * 100)).append("%)\n");
                }
            }

            final String resStr = sb.length() > 0 ? sb.toString() : "Waiting for detections...";
            final int count = latestBoxes.length;

            runOnUiThread(() -> {
                if (resultTextView != null) {
                    resultTextView.setText(resStr);
                }
                if (detectionCountChip != null) {
                    detectionCountChip.setText(count + " objects");
                }
            });
        }

        holder.unlockCanvasAndPost(canvas);
    }

    // -- Barebones GL Boilerplate for ARCore Background --
    private void setupGlQuad() {
        float[] vertices = {
                -1.0f, -1.0f, 0.0f,
                1.0f, -1.0f, 0.0f,
                -1.0f, 1.0f, 0.0f,
                1.0f, 1.0f, 0.0f,
        };
        float[] texCoords = {
                0.0f, 1.0f,
                1.0f, 1.0f,
                0.0f, 0.0f,
                1.0f, 0.0f,
        };

        ByteBuffer bbVerts = ByteBuffer.allocateDirect(vertices.length * 4);
        bbVerts.order(ByteOrder.nativeOrder());
        quadVertices = bbVerts.asFloatBuffer();
        quadVertices.put(vertices);
        quadVertices.position(0);

        ByteBuffer bbTex = ByteBuffer.allocateDirect(texCoords.length * 4);
        bbTex.order(ByteOrder.nativeOrder());
        quadTexCoord = bbTex.asFloatBuffer();
        quadTexCoord.put(texCoords);
        quadTexCoord.position(0);

        ByteBuffer bbTexT = ByteBuffer.allocateDirect(texCoords.length * 4);
        bbTexT.order(ByteOrder.nativeOrder());
        quadTexCoordTransformed = bbTexT.asFloatBuffer();
        quadTexCoordTransformed.put(texCoords); // Default fallback
        quadTexCoordTransformed.position(0);
    }

    private void compileGlProgram() {
        String vertexShader = "attribute vec4 a_Position;" +
                "attribute vec2 a_TexCoord;" +
                "varying vec2 v_TexCoord;" +
                "void main() {" +
                "   gl_Position = a_Position;" +
                "   v_TexCoord = a_TexCoord;" +
                "}";

        String fragmentShader = "#extension GL_OES_EGL_image_external : require\n" +
                "precision mediump float;" +
                "varying vec2 v_TexCoord;" +
                "uniform samplerExternalOES sTexture;" +
                "void main() {" +
                "  gl_FragColor = texture2D(sTexture, v_TexCoord);" +
                "}";

        int vertexShaderId = GLES20.glCreateShader(GLES20.GL_VERTEX_SHADER);
        GLES20.glShaderSource(vertexShaderId, vertexShader);
        GLES20.glCompileShader(vertexShaderId);

        int fragmentShaderId = GLES20.glCreateShader(GLES20.GL_FRAGMENT_SHADER);
        GLES20.glShaderSource(fragmentShaderId, fragmentShader);
        GLES20.glCompileShader(fragmentShaderId);

        quadProgram = GLES20.glCreateProgram();
        GLES20.glAttachShader(quadProgram, vertexShaderId);
        GLES20.glAttachShader(quadProgram, fragmentShaderId);
        GLES20.glLinkProgram(quadProgram);

        quadPositionParam = GLES20.glGetAttribLocation(quadProgram, "a_Position");
        quadTexCoordParam = GLES20.glGetAttribLocation(quadProgram, "a_TexCoord");
        quadTextureParam = GLES20.glGetUniformLocation(quadProgram, "sTexture");
    }

    private void drawBackground() {
        GLES20.glUseProgram(quadProgram);
        GLES20.glEnableVertexAttribArray(quadPositionParam);
        GLES20.glEnableVertexAttribArray(quadTexCoordParam);

        // ALWAYS Rewind buffers before reading into GL to avoid blank geometry
        quadVertices.position(0);
        quadTexCoordTransformed.position(0);

        GLES20.glVertexAttribPointer(quadPositionParam, 3, GLES20.GL_FLOAT, false, 0, quadVertices);
        GLES20.glVertexAttribPointer(quadTexCoordParam, 2, GLES20.GL_FLOAT, false, 0, quadTexCoordTransformed);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);
        GLES20.glUniform1i(quadTextureParam, 0);

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

        GLES20.glDisableVertexAttribArray(quadPositionParam);
        GLES20.glDisableVertexAttribArray(quadTexCoordParam);
    }
}
