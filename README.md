```html
<!DOCTYPE html>
<html>
<head>
  <title>Image Classificaation in Spark</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
  <style>
  body {
      position: relative; 
  }
  #section1 {padding-top:50px;height:200px;}
  #section2 {padding-top:50px;height:800px;}
  #section3 {padding-top:50px;height:1500px;}
  #section4 {padding-top:150px;height:2500px;}
  #section4 {padding-top:150px;height:500px;}
  </style>
  <script type="text/javascript">
    $(document).ready(function(){
       $("#knn-table").hide();
      
      $("#knnbtn").click(function(){
        $("#knn-table").show();
        $("#neural-table").hide();
      });


      $("#neuralbtn").click(function(){
        $("#knn-table").hide();
        $("#neural-table").show();
      })
    })
  </script>
</head>
<body data-spy="scroll" data-target=".navbar" data-offset="50">

<nav class="navbar navbar-inverse navbar-fixed-top">
  <div class="container-fluid">
    <div class="navbar-header">
        <button type="button" class="navbar-toggle" data-toggle="collapse" data-target="#myNavbar">
          <span class="icon-bar"></span>
          <span class="icon-bar"></span>
          <span class="icon-bar"></span>                        
      </button>
      <a class="navbar-brand" href="#">Cloud Computing</a>
    </div>
    <div>
      <div class="collapse navbar-collapse" id="myNavbar">
        <ul class="nav navbar-nav">
          <li><a href="#section1">Introduction</a></li>
          <li><a href="#section2">Dataset</a></li>
          <li><a href="#section3">Team Members</a></li>
          <li><a href="#section4">Preprocessing</a></li>
          <li><a href="#section5">Algorithms</a></li>
        </ul>
      </div>
    </div>
  </div>
</nav>    

<div id="section1" class="container-fluid">
  <h1 class="col-md-offset-4">Distracted Driver Detection</h1>
 <div class="col-md-offset-2 col-md-8 well">
    <p>According to the CDC motor vehicle safety division, one in five car accidents is caused by a distracted driver.</p>
    <p>Given a dataset of 2D dashboard camera images, Kaggle is providing the dataset to classify each driver image.</p>
    <p>For our final project, we decided to work on predicting whether an image can be classified into different categories with Apache Spark using OpenCV.</p>
 </div>
</div>

<div id="section2" class="container-fluid">
  <h1 class="col-md-offset-5">Dataset</h1>
  <div class="col-md-offset-2 col-md-8 well">
    <p>We got this dataset from Kaggle. Link is here (<a href="https://www.kaggle.com/c/state-farm-distracted-driver-detection">Take me to the Kaggle.</a>).</p>
    <p>Input data has 10 categories where each category has images shows driver is doing something.</p>
    <p class="col-md-offset-4">The 10 classes in the training set are:</p>
    <ol class="col-md-offset-4">
      <li>c0: safe driving</li>
      <li>c1: texting - right</li>
      <li>c2: talking on the phone - right</li>
      <li>c3: texting - left</li>
      <li>c4: talking on the phone - left</li>
      <li>c5: operating the radio</li>
      <li>c6: drinking</li>
      <li>c7: reaching behind</li>
      <li>c8: hair and makeup</li>
      <li>c9: talking to passenger</li>
    </ol> 
    <p>Our goal is to predict what the driver is doing in each picture</p>
    <img class="col-md-offset-4" src="data.gif" class="img-rounded" height="300">
  </div>
</div>

<div id="section3" class="container-fluid">
  <h1 class="col-md-offset-5">Project Group</h1>

  <div class="card col-md-4">
  <img class="card-img-top col-md-offset-3" src="saikalyan.png" alt="Card image cap" width="150px" height="170px">
  <div class="card-block">
    <h4 class="card-title">Saikalyan Yeturu</h4>
    <div class="card-text col-md-8">
       <div class="progress">
      <div class="progress-bar" role="progressbar" aria-valuenow="50"
      aria-valuemin="0" aria-valuemax="100" style="width:40%">
        Preprocessing 25%
      </div>
    </div>

    <div class="progress">
      <div class="progress-bar" role="progressbar" aria-valuenow="25"
      aria-valuemin="0" aria-valuemax="100" style="width:40%">
        KNN 40%
      </div>
    </div>

    <div class="progress">
      <div class="progress-bar" role="progressbar" aria-valuenow="40"
      aria-valuemin="0" aria-valuemax="100" style="width:40%">
        NeuralNetwork 30%
      </div>
    </div>

    <div class="progress">
      <div class="progress-bar" role="progressbar" aria-valuenow="30"
      aria-valuemin="0" aria-valuemax="100" style="width:30%">
        Evaluation 30%
      </div>
    </div>
    </div>
  </div>
</div>

 <div class="card col-md-4">
  <img class="card-img-top col-md-offset-3" src="vamsi.jpg" alt="Card image cap" width="150px" height="170px">
  <div class="card-block">
    <h4 class="card-title">Vamsi Krishna Kovuru</h4>
    <div class="card-text col-md-8">

      <div class="progress">
      <div class="progress-bar" role="progressbar" aria-valuenow="50"
      aria-valuemin="0" aria-valuemax="100" style="width:50%">
        Preprocessing 50%
      </div>
    </div>

    <div class="progress">
      <div class="progress-bar" role="progressbar" aria-valuenow="25"
      aria-valuemin="0" aria-valuemax="100" style="width:30%">
        KNN 30%
      </div>
    </div>

    <div class="progress">
      <div class="progress-bar" role="progressbar" aria-valuenow="40"
      aria-valuemin="0" aria-valuemax="100" style="width:40%">
        NeuralNetwork 40%
      </div>
    </div>

    <div class="progress">
      <div class="progress-bar" role="progressbar" aria-valuenow="30"
      aria-valuemin="0" aria-valuemax="100" style="width:40%">
        Evaluation 40%
      </div>
    </div>

    </div>
  </div>
</div>

 <div class="card col-md-4">
  <img class="card-img-top col-md-offset-3" src="vivek.jpg" alt="Card image cap"width="150px" height="170px">
  <div class="card-block">
    <h4 class="card-title">Vivekananda Adepu</h4>
    <div class="card-text col-md-8">
         <div class="progress">
      <div class="progress-bar" role="progressbar" aria-valuenow="50"
      aria-valuemin="0" aria-valuemax="100" style="width:40%">
        Preprocessing 25%
      </div>
    </div>

    <div class="progress">
      <div class="progress-bar" role="progressbar" aria-valuenow="25"
      aria-valuemin="0" aria-valuemax="100" style="width:30%">
        KNN 30%
      </div>
    </div>

    <div class="progress">
      <div class="progress-bar" role="progressbar" aria-valuenow="40"
      aria-valuemin="0" aria-valuemax="100" style="width:40%">
        NeuralNetwork 30%
      </div>
    </div>

    <div class="progress">
      <div class="progress-bar" role="progressbar" aria-valuenow="30"
      aria-valuemin="0" aria-valuemax="100" style="width:30%">
        Evaluation 30%
      </div>
    </div>
  </div>
</div>

</div>


<div id="section4" class="container-fluid">
  <h1 class="col-md-offset-5">Preprocessing</h1>
 <div class="col-md-offset-1 col-md-10 well">
    <p>Applying machine learning algorithms directly on raw images does not give positive results. So images must be pre-processes before using them for training.</p>
    <p>Some of the pre-processing steps for any type of image classification are mentioned below.</p>
    <div class="row">
      <div class="col-md-6">
          <p><span style="font-size: 15px" class="label label-default">Original Image given in the Dataset:</span></p>
          <img height="250" width="350" src="project_images/orig.jpg">
      </div>
          <div class="col-md-6">
         <p><span style="font-size: 15px" class="label label-default">Aplying Grey scale and Bulrring techniques to the images:</span></p>
          <img height="250" width="350" src="project_images/original.jpg">
      </div><br><br>
      <div class="col-md-6">
        <p><span style="font-size: 15px" class="label label-default">Aplying Edge Detection to the images:</span></p>
          <img height="250" width="350" src="project_images/edge.jpg">
      </div>
      <div class="col-md-6">
        <p><span style="font-size: 15px" class="label label-default">Using key point descriptors to identify edges.</span></p>
          <img height="250" width="350" src="project_images/key.jpg">
      </div><br>
      <div class="col-md-6">
        <p><span style="font-size: 15px" class="label label-default">Cropping the images.</span></p>
          <img height="250" width="350" src="project_images/Cropped.jpg">
      </div>
      <div class="col-md-6">
        <p><span style="font-size: 15px" class="label label-default">Applying Fore-ground reduction techniques to the images.</span></p>
          <img height="250" width="350" src="project_images/Fore.jpg">
      </div>

    </div>

    <h1 class="col-md-offset-4">Feature Engineering:</h1>
    <p>Main challange involved in image classification is <strong>Feature Engineering.</strong></p>
    <p>The dataset on which we have worked is about detecting whether is driver is distracted or not. So face and hands of the driver are most important features of the image. </p>
    <p>Some of the methods used for the Feature Engineering are mentioned below.</p>
    
    <div class="row col-md-offset-1">
        <div class="col-md-6">
          <p>Applying Threshold Segmentation to extract Face and hands.</p>
          <img src="project_images/segmentation.png">
        </div>
        <div class="col-md-6">
          <p>Resizing the image to 50 x 50 for the easy processing.</p>
     <img src="project_images/resize.png">
        </div>
    </div>
 </div>
</div>

<div id="section5" class="container-fluid">
  <h1 class="col-md-offset-4">Algorithms Implementation:</h1>

 <div class="col-md-offset-2 col-md-8 well">
    <h2 class="col-md-offset-4">Neural Networks:</h2>
    <p></p>
 </div>

  <div class="col-md-offset-2 col-md-8 well">
    <h2 class="col-md-offset-4">K-Nearest Neighbours:</h2>
 </div>
</div>

<div id="section6" class="container-fluid">
  <h1 class="col-md-offset-6">Results:</h1>

 <div class="col-md-offset-2 col-md-8 well">
  <div class="col-md-offset-4">
    <button id="knnbtn" class="btn btn-primary">Show KNN results</button>
     <button id="neuralbtn" class="btn btn-primary">Show Neural Network results</button>
  </div>
   <div id="knn-table" class="col-md-offset-5">
    <h2>KNN results:</h2>
    <img src="project_images/knn.jpg">
 </div>

 <div id="neural-table" class="col-md-offset-4">
    <h2>Neural Network results:</h2>
    <img src="project_images/neural.jpg">
 </div>
 </div>
</div>


</body>
</html>

```