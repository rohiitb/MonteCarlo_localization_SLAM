# MonteCarlo_localization_SLAM

This project was done as a part of the course ESE650 : Learning  in Robotics at UPenn. This project involves implementing Simultaneous Localization and Mapping based on Odometry data and 2D-Lidar scans. It uses MonteCarlo sampling and Particle filter to get the localization and generate the occupancy grid of the environment. The pipeline is shown below:

<p float="center">
  <img src="./Results/PF_diag.jpg" alt="Algorithm" class="center">
</p>


# Results

<table>
  <tr>
      <td align = "center"> <img src="./Results/map1.png" /> </td>
      <td align = "center"> <img src="./Results/map2.png" /> </td>
  </tr>
  <tr>
      <td align = "center"> World 1 </td>
      <td align = "center"> World 2 </td>
  </tr>
</table>

<table>
  <tr>
      <td align = "center"> <img src="./Results/map3.png" /> </td>
      <td align = "center"> <img src="./Results/map4.png" /> </td>
  </tr>
  <tr>
      <td align = "center"> World 3 </td>
      <td align = "center"> World 4 </td>
  </tr>
</table>
