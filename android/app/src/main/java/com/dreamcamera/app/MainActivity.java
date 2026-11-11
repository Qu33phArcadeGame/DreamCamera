package com.dreamcamera.app;

import android.Manifest;
import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Base64;
import android.view.WindowManager;
import android.webkit.JavascriptInterface;
import android.webkit.PermissionRequest;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import java.io.OutputStream;

// DreamCamera — WebView wrapper of the web app in assets/.
// The page loads from file:///android_asset/ with file-URL access allowed,
// so it can fetch the bundled models/ folder and the live model URLs.
// getUserMedia needs the CAMERA runtime permission plus a
// WebChromeClient that grants VIDEO_CAPTURE to the page.
// Home screen widgets (DreamWidgetProvider 2x2, DreamWidgetLargeProvider
// 3x3) show the latest dream frame; the page hands each finished dream
// to the DreamCameraBridge below, which saves it and refreshes them.
public class MainActivity extends Activity {
    private static final int REQ_CAMERA = 1;
    private static final int REQ_FILE_CHOOSER = 2;
    private WebView webView;
    private ValueCallback<Uri[]> filePathCallback;

    // Called from the web app when a dream run finishes: the page passes
    // the latest frame as a data: URL. Saved for the widgets to show.
    private class DreamBridge {
        @JavascriptInterface
        public void setLatestDream(String dataUrl) {
            DreamWidgetHelper.saveLatestDream(MainActivity.this, dataUrl);
        }

        // Called from the web app when the user saves a file (a dreamed
        // video or a still): the page passes it as a data: URL. A WebView
        // ignores the download attribute on its own, so the page hands the
        // file to the app instead. Saved to Downloads via MediaStore
        // (Android 10+) so it shows up in Files.
        @JavascriptInterface
        public void saveFile(String dataUrl, String filename, String mime) {
            try {
                if (dataUrl == null || Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return;
                int comma = dataUrl.indexOf(',');
                if (comma < 0) return;
                byte[] bytes = Base64.decode(dataUrl.substring(comma + 1), Base64.DEFAULT);
                if (mime == null || mime.isEmpty()) mime = "application/octet-stream";
                ContentValues v = new ContentValues();
                v.put(MediaStore.MediaColumns.DISPLAY_NAME, filename);
                v.put(MediaStore.MediaColumns.MIME_TYPE, mime);
                v.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS);
                Uri uri = getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, v);
                if (uri == null) return;
                OutputStream out = getContentResolver().openOutputStream(uri);
                if (out == null) return;
                out.write(bytes);
                out.close();
            } catch (Exception e) {
                // A failed save must never break the app.
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        webView = new WebView(this);
        setContentView(webView);
        WebSettings s = webView.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setDatabaseEnabled(true);
        s.setAllowFileAccess(true);
        s.setAllowFileAccessFromFileURLs(true);
        s.setAllowUniversalAccessFromFileURLs(true);
        s.setMediaPlaybackRequiresUserGesture(false);
        webView.setWebViewClient(new WebViewClient());
        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(PermissionRequest request) {
                // The page asks for VIDEO_CAPTURE for getUserMedia.
                // Grant it only if the app itself holds the camera permission.
                if (checkSelfPermission(Manifest.permission.CAMERA)
                        == PackageManager.PERMISSION_GRANTED) {
                    request.grant(request.getResources());
                } else {
                    request.deny();
                }
            }

            @Override
            public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> callback,
                                             FileChooserParams params) {
                // The page opens a file input (photo upload, style photo, or
                // video). A WebView shows no picker on its own, so open one
                // here — with the MIME type the page asked for. The old code
                // hardcoded image/*, so the video picker only showed photos.
                if (filePathCallback != null) filePathCallback.onReceiveValue(null);
                filePathCallback = callback;
                String type = "image/*";
                String title = "Choose a photo";
                String[] accept = params.getAcceptTypes();
                if (accept != null && accept.length > 0 && accept[0] != null && !accept[0].isEmpty()) {
                    type = accept[0];
                    if (type.startsWith("video")) title = "Choose a video";
                }
                Intent i = new Intent(Intent.ACTION_GET_CONTENT);
                i.addCategory(Intent.CATEGORY_OPENABLE);
                i.setType(type);
                startActivityForResult(Intent.createChooser(i, title), REQ_FILE_CHOOSER);
                return true;
            }
        });
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
                && checkSelfPermission(Manifest.permission.CAMERA)
                        != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQ_CAMERA);
        }
        webView.addJavascriptInterface(new DreamBridge(), "DreamCameraBridge");
        // A widget's Shoot & Dream button launches with autodream: the page
        // starts a dream by itself once the camera is ready (see app.js).
        boolean autodream = getIntent() != null && getIntent().getBooleanExtra("autodream", false);
        webView.loadUrl("file:///android_asset/index.html" + (autodream ? "?autodream=1" : ""));
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        // The app is already open and a widget tap brought it forward:
        // only a Shoot & Dream tap reloads the page to start a dream.
        if (intent != null && intent.getBooleanExtra("autodream", false)) {
            webView.loadUrl("file:///android_asset/index.html?autodream=1");
        }
    }

    @Override
    protected void onActivityResult(int code, int result, Intent data) {
        super.onActivityResult(code, result, data);
        if (code == REQ_FILE_CHOOSER && filePathCallback != null) {
            Uri[] results = null;
            if (result == RESULT_OK && data != null && data.getData() != null) {
                results = new Uri[]{data.getData()};
            }
            filePathCallback.onReceiveValue(results);
            filePathCallback = null;
        }
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] perms, int[] results) {
        super.onRequestPermissionsResult(code, perms, results);
        // The page re-asks getUserMedia when the user shoots; a reload
        // makes the camera start cleanly after the grant.
        if (code == REQ_CAMERA) webView.reload();
    }

    @Override
    public void onBackPressed() {
        if (webView.canGoBack()) webView.goBack();
        else super.onBackPressed();
    }

    @Override
    protected void onDestroy() {
        if (webView != null) webView.destroy();
        super.onDestroy();
    }
}
