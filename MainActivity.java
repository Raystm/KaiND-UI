package com.PugOrBulldog.android_with_opencv;

import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.DecimalFormat;
import java.util.Date;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    /**カメラパラメータ**/
    private static final String TAG = "MyApp";
    private int REQUEST_CODE_FOR_PERMISSIONS = 1234;;
    private final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA"};

    /**ビューの用意**/
    private PreviewView previewView;
    private ImageView imageView;

    /**カメラのメンバ**/
    private Camera camera = null;
    private Preview preview = null;
    private ImageAnalysis imageAnalysis = null;
    private ExecutorService cameraExecutor = Executors.newSingleThreadExecutor();

    /** Tensor Flowについての情報を格納したメンバ **/
    String ModelPath;
    Context context;
    MappedByteBuffer myModel;

    static
    {
        System.loadLibrary("opencv_java4");
    }

    /****************************************
     *　TensorFlowのモデル取得関数
     * **************************************/
    private MappedByteBuffer loadModel(Context context, String modelPath)
    {
        try
        {
            AssetFileDescriptor fd = context.getAssets().openFd(modelPath);
            FileInputStream in = new FileInputStream(fd.getFileDescriptor());
            FileChannel fileChannel = in.getChannel();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY,
                    fd.getStartOffset(), fd.getDeclaredLength());
        }
        catch (Exception e)
        {
            e.printStackTrace();
            return null;
        }
    }

    /****************************************
     * 起動時に呼ばれる関数
     * andoroidの起動プロセスに組み込まれているメソッドをオーバライドすることで任意の処理を追加している。
     * ビューやボタンに対数するイベントハンドラをここで設定
     ****************************************/
    @Override
    protected void onCreate(Bundle savedInstanceState)
    {

        super.onCreate(savedInstanceState);
            setContentView(R.layout.activity_main);

            /**ビューの設定**/
            previewView = findViewById(R.id.previewView);
            imageView = findViewById(R.id.imageView);




            if (checkPermissions())
            {

                /**  TesnsorFlowの読み込み**/
//                ModelPath = "model_img_recog_pug_bull_bn.tflite";
                ModelPath = "model_img_recog_pug_bull_FT_byTFhub.tflite";
                context = MainActivity.this;
                myModel = loadModel(context, ModelPath);

                /****************************
                 * カメラ設定メイン関数
                 ****************************/
                startCamera();

            }
            else
            {
                ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_FOR_PERMISSIONS);
            }
    }

    /****************************************
     * カメラの設定(フームごとの処理などをここで設置)
     ****************************************/
    private void startCamera()
    {

        /**カメラ取得**/
        final ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        Context context = this;

        cameraProviderFuture.addListener(new Runnable()
        {
            /**カメラ内のフレームごとの処理をオーバーライド, 画像に関して毎フレーム処理を行う関数を設定する**/
            @Override
            public void run()
            {
                try
                {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    preview = new Preview.Builder().build();

                    /**毎フレームの画像処理実行**/
                    imageAnalysis = new ImageAnalysis.Builder().build();
                    imageAnalysis.setAnalyzer(cameraExecutor, new MyImageAnalyzer());

                    CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build();
                    cameraProvider.unbindAll();
                    camera = cameraProvider.bindToLifecycle((LifecycleOwner)context, cameraSelector, preview, imageAnalysis);
                    preview.setSurfaceProvider(previewView.createSurfaceProvider(camera.getCameraInfo()));
                }
                catch(Exception e)
                {
                    Log.e(TAG, "[startCamera] Use case binding failed", e);
                }
            }
        }, ContextCompat.getMainExecutor(this));
    }

    /****************************************
     * 毎フレームの画像処理を管理するクラス
     ****************************************/
    private class MyImageAnalyzer implements ImageAnalysis.Analyzer
    {
        /**識別時の情報**/
        final int CLASS_NUM = 3;
        final int IMG_WIDTH = 224;
        final int IMG_HEIGHT = 224;

        private long pre_recog_time = 0;
        private long inferenceTime = 0;
        private int count = 0;

        /**識別結果値格納配列**/
        float[][] output = new float[1][CLASS_NUM];

        /**識別クラス**/
        private String STR_CLASS_0 = "Pug";
        private String STR_CLASS_1 = "Bulldog";
        private String STR_CLASS_2 = "Other";
        private String STR_CLASS_3 = "Speed";

        private Interpreter interpreter = new Interpreter(myModel);

        TextView category0 = findViewById(R.id.category0);
        TextView category1 = findViewById(R.id.category1);
        TextView category2 = findViewById(R.id.category2);
        TextView classify_speed = findViewById(R.id.classify_speed);

        /**********************************************
         **カメラ画像を引数とするメソッドをオーバーライド*
         **********************************************/
        @Override
        public void analyze(@NonNull ImageProxy image)
        {
            /**カメラ画像をopencvで取得(BGR)**/
            Mat matFromCamera = getMatFromImage(image);

            /**推論入力画像(RGB)を作成**/
            Mat matRead = new Mat();
            Imgproc.resize(matFromCamera, matRead, new Size(IMG_WIDTH, IMG_HEIGHT), 0,0, Imgproc.INTER_LINEAR);
            Imgproc.cvtColor(matRead, matRead, Imgproc.COLOR_BGR2RGB, 3);

            /**出力用画像(BGR)を用意**/
            Mat matOutput = new Mat();
            Imgproc.resize(matFromCamera, matOutput, new Size(IMG_WIDTH, IMG_HEIGHT), 0,0, Imgproc.INTER_LINEAR);
            //Imgproc.cvtColor(matOutput, matOutput, Imgproc.COLOR_BGR2RGB, 3);

            /** 1秒間隔で推論が行われるようにするため、時刻を取得 **/
            final Date date = new Date();
            if (pre_recog_time == 0)
            {
                pre_recog_time = date.getTime();
                Log.i(TAG, "FIRST pre_recog_time = " + pre_recog_time);
            }

            /******************************
             * 3秒ごとに実施される推論処理
             ******************************/
            if (3000 <= date.getTime() - pre_recog_time)
            {
                {
                    /*****************************************/
                    /*********** 推論の実施 *******************/
                    /*****************************************/
                    System.out.println("Into interpreter");
                    /* [0,255]の画像を[0.0, 1.0]に正規化 */
                    float[][][][] testImg = new float[1][IMG_HEIGHT][IMG_WIDTH][3];
                    for (int i = 0; i < IMG_WIDTH; i++)
                    {
                        for (int j = 0; j < IMG_HEIGHT; j++)
                        {
                            double[] value = matRead.get(j, i);

//                            testImg[0][j][i][0] = (float) value[0] / 255.0f;
//                            testImg[0][j][i][1] = (float) value[1] / 255.0f;
//                            testImg[0][j][i][2] = (float) value[2] / 255.0f;
                            /**TensorFlowHubに合わせてRGBの順番で入れる**/
                            testImg[0][j][i][0] = (float) value[2] / 255.0f;
                            testImg[0][j][i][1] = (float) value[1] / 255.0f;
                            testImg[0][j][i][2] = (float) value[0] / 255.0f;
                        }
                    }

                    /**推論実施**/
                    long inferenceStartTime = System.currentTimeMillis();
                    interpreter.run(testImg, output);
                    inferenceTime = System.currentTimeMillis() - inferenceStartTime;
                    Log.i(TAG, "inferenceStartTime = " + inferenceStartTime);
                    Log.i(TAG, "Speed = " + inferenceTime);
//                    output[0][0] = 0.0f;
//                    output[0][1] = 0.7f;
//                    output[0][2] = 0.0f;

                    /*********** デバッグ処理ここから *********/
//                    Mat matReadDebug = new Mat();
//                    Imgproc.cvtColor(matRead, matReadDebug, Imgproc.COLOR_BGR2RGB, 3);
//                    final long currentTimeMillis = System.currentTimeMillis ();
//                    final String appName = getString (R.string.app_name);
//                    final String galleryPath =
//                            Environment.getExternalStoragePublicDirectory (
//                                    Environment.DIRECTORY_PICTURES) .toString ();
//                    final String albumPath = galleryPath + "/" + appName;
//                    final String photoName = "img_" + String.format("%06d", count) + ".png";
//                    final String photoPath = albumPath + "/" + photoName;
//                    final ContentValues values = new ContentValues();
//                    // ファイル名
//                    values.put(MediaStore.Images.Media.DISPLAY_NAME, photoName);
//                    values.put(MediaStore.MediaColumns.DATA, photoPath);
//                    values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
//                    values.put(MediaStore.Images.Media.TITLE, appName);
//                    values.put(MediaStore.Images.Media.DESCRIPTION, appName);
//                    values.put(MediaStore.Images.Media.DATE_TAKEN, currentTimeMillis);
//                    // 書込み時にメディア ファイルに排他的にアクセスする
//                    values.put(MediaStore.Images.Media.IS_PENDING, 1);
//
//                    File album = new File (albumPath);
//                    if (! album.isDirectory () &&! album.mkdirs ())
//                    {
//                        Log.e (TAG, "Failed to create album directory at" +
//                                albumPath);
//                    }
//                    Log.i(TAG, "photoPath = " + photoPath);
//                    boolean isSaved = Imgcodecs.imwrite(photoPath, matReadDebug);
//                    Log.i(TAG, "isSaved = " + String.valueOf(isSaved));
//
//                    values.clear();
//                    //　排他的にアクセスの解除
//                    values.put(MediaStore.Images.Media.IS_PENDING, 0);
                    /*********** デバッグ処理ここままで *********/


                }

                count = count + 1;

                pre_recog_time = date.getTime();
            }


            /**BitMapの仕様と合わせるため、行列を転置**/
            matOutput = matOutput.t();
            //Imgproc.cvtColor(matOutput, matOutput, Imgproc.COLOR_BGR2RGB, 3);
            Core.flip(matOutput, matOutput, 1);
            Imgproc.resize(matOutput, matOutput, new Size(1000, 1400));

            /* 結果の描画 */
//            String result_0 = String.valueOf((int)(output[0][0]*100)) + "%";
//            String result_1 = String.valueOf((int)(output[0][1]*100)) + "%";
            float temp = 1- (output[0][0] + output[0][1]);
//            String result_2 = String.valueOf((int)(temp)*100) + "%";

            String result_0 = new DecimalFormat("00.0").format (output[0][0]*100)  + "%";
            String result_1 = new DecimalFormat("00.0").format (output[0][1]*100)  + "%";
            String result_2 = new DecimalFormat("00.0").format (temp*100)  + "%";
            String result_3 = new DecimalFormat("00.0").format (inferenceTime)  + "[ms]";


            /**推論結果を画像に書き込み**/
            Scalar red = new Scalar(255, 0, 0);
            Scalar gray = new Scalar(180, 180, 180);
//            if(0.5 < output[0][0])
//            {
//                /*最大がラーメン*/
//                Imgproc.putText(matOutput, STR_CLASS_0, new Point(20, 80), 3, 2.5, red, 3);
//                Imgproc.putText(matOutput, result_0, new Point(520, 80), 3, 2.5, red, 3);
//
//                Imgproc.putText(matOutput, STR_CLASS_1, new Point(20, 170), 3, 2.5, gray, 3);
//                Imgproc.putText(matOutput, result_1, new Point(520, 170), 3, 2.5, gray, 3);
//
//                Imgproc.putText(matOutput, STR_CLASS_2, new Point(20, 260), 3, 2.5, gray, 3);
//                Imgproc.putText(matOutput, result_2, new Point(520, 260), 3, 2.5, gray, 3);
//
//                Imgproc.putText(matOutput, STR_CLASS_3, new Point(20, 350), 3, 2.5, gray, 3);
//                Imgproc.putText(matOutput, result_3, new Point(520, 350), 3, 2.5, gray, 3);
//
//
//
//            }
//            else if(0.5 < output[0][1])
//            {
//                /*最大が海藻*/
//                Imgproc.putText(matOutput, STR_CLASS_0, new Point(20, 80), 3, 2.5, gray, 3);
//                Imgproc.putText(matOutput, result_0, new Point(520, 80), 3, 2.5, gray, 3);
//
//                Imgproc.putText(matOutput, STR_CLASS_1, new Point(20, 170), 3, 2.5, red, 3);
//                Imgproc.putText(matOutput, result_1, new Point(520, 170), 3, 2.5, red, 3);
//
//                Imgproc.putText(matOutput, STR_CLASS_2, new Point(20, 260), 3, 2.5, gray, 3);
//                Imgproc.putText(matOutput, result_2, new Point(520, 260), 3, 2.5, gray, 3);
//
//                Imgproc.putText(matOutput, STR_CLASS_3, new Point(20, 350), 3, 2.5, gray, 3);
//                Imgproc.putText(matOutput, result_3, new Point(520, 350), 3, 2.5, gray, 3);
//            }
//            else if(0.5 < 1.0 - (output[0][0] + output[0][1]))
//            {
//                /*最大がその他*/
//                Imgproc.putText(matOutput, STR_CLASS_0, new Point(20, 80), 3, 2.5, gray, 3);
//                Imgproc.putText(matOutput, result_0, new Point(520, 80), 3, 2.5, gray, 3);
//
//                Imgproc.putText(matOutput, STR_CLASS_1, new Point(20, 170), 3, 2.5, gray, 3);
//                Imgproc.putText(matOutput, result_1, new Point(520, 170), 3, 2.5, gray, 3);
//
//                Imgproc.putText(matOutput, STR_CLASS_2, new Point(20, 260), 3, 2.5, red, 3);
//                Imgproc.putText(matOutput, result_2, new Point(520, 260), 3, 2.5, red, 3);
//
//                Imgproc.putText(matOutput, STR_CLASS_3, new Point(20, 350), 3, 2.5, gray, 3);
//                Imgproc.putText(matOutput, result_3, new Point(520, 350), 3, 2.5, gray, 3);
//            }
            category0.setText(result_0);
            category1.setText(result_1);
            category2.setText(result_2);
            classify_speed.setText(result_3);

//            category0.setText(STR_CLASS_0 + result_0);
//            category1.setText(STR_CLASS_1 + result_1);
//            category2.setText(STR_CLASS_2 + result_2);
//            classify_speed.setText(STR_CLASS_3 + result_3);

            /**画像処理した行列をBitMapに変換**/
            Bitmap bitmap = Bitmap.createBitmap(matOutput.cols(), matOutput.rows(),Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(matOutput, bitmap);

            /** BitMapをビューに設定 **/
            runOnUiThread(new Runnable()
            {
                @Override
                public void run()
                {
                    imageView.setImageBitmap(bitmap);
                }
            });

            /* Close the image otherwise, this function is not called next time */
            image.close();
        }

        /****************************************
         * OpenCvの画像を取得関数
         ****************************************/
        private Mat getMatFromImage(ImageProxy image)
        {
            /* https://stackoverflow.com/questions/30510928/convert-android-camera2-api-yuv-420-888-to-rgb */
            ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
            ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
            ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();
            int ySize = yBuffer.remaining();
            int uSize = uBuffer.remaining();
            int vSize = vBuffer.remaining();
            byte[] nv21 = new byte[ySize + uSize + vSize];
            yBuffer.get(nv21, 0, ySize);
//            vBuffer.get(nv21, ySize, vSize);
//            uBuffer.get(nv21, ySize + vSize, uSize);
            uBuffer.get(nv21, ySize, uSize);
            vBuffer.get(nv21, ySize + uSize, vSize);
            Mat yuv = new Mat(image.getHeight() + image.getHeight() / 2, image.getWidth(), CvType.CV_8UC1);
            yuv.put(0, 0, nv21);
            Mat mat = new Mat();
            //Imgproc.cvtColor(yuv, mat, Imgproc.COLOR_YUV2BGR_I420, 3);
            Imgproc.cvtColor(yuv, mat, Imgproc.COLOR_YUV2BGR_NV21, 3);

            return mat;
        }
    }

    /****************************************
     * TensorFlowの読み込みチェック
     ****************************************/
    private boolean checkPermissions()
    {
        for(String permission : REQUIRED_PERMISSIONS)
        {
            if(ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED)
            {
                return false;
            }
        }
        return true;
    }

}