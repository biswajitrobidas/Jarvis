# Jarvis import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.common.model.Model;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    // Replace with the actual path to your TensorFlow Lite model file
    private static final String MODEL_PATH = "path/to/your/model.tflite"; 

    private ImageView imageView;
    private Button predictButton;
    private TextView resultTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        predictButton = findViewById(R.id.predictButton);
        resultTextView = findViewById(R.id.resultTextView);

        predictButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                predictColor();
            }
        });
    }

    private void predictColor() {
        try {
            // Load the TensorFlow Lite model
            Model model = Model.createModel(this, MODEL_PATH);

            // Load the image
            Bitmap image = BitmapFactory.decodeResource(getResources(), R.drawable.your_image); 

            // Preprocess the image
            TensorImage inputImage = TensorImage.fromBitmap(image);

            // Create input and output tensors
            ByteBuffer inputBuffer = inputImage.getBuffer();
            TensorBuffer outputBuffer = TensorBuffer.createFixedSize(new int[]{1, 10}, DataType.FLOAT32); 

            // Run inference
            model.run(inputBuffer, outputBuffer.getBuffer());

            // Get the top-k results
            List<TensorLabel> labels = TensorLabel.fromTensorBuffer(outputBuffer, model.getOutputLabels());

            // Display the result
            String result = "Predicted Color: " + labels.get(0).getLabel();
            resultTextView.setText(result);

        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Error loading model", Toast.LENGTH_SHORT).show();
        }
    }
}
