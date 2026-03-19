# PA-LVIO: Real-Time LiDAR-Visual-Inertial Odometry and Mapping with Pose-Only Bundle Adjustment

We propose a novel LiDAR-Visual-Inertial Odometry (LVIO) for real-time navigation and mapping. PA-LVIO employs a pose-only bundle adjustment (PA) framework, which is highly accurate and efficient, for both frame-to-frame (F2F) LiDAR and visual measurements. Meanwhile, a marginalization-free and frame-to-map (F2M) LiDAR measurement model is integrated into PA-LVIO to eliminate odometry drifts. Besides, an IMU-centric online spatial-temporal calibration is employed to obtain a pixel-wise LiDAR-camera alignment for high-quality and RGB-rendered mapping.

**Author:** Hailiang Tang from the [Integrated and Intelligent Navigation (i2Nav) Group](http://www.i2nav.com/), GNSS Research Center, Wuhan University.

**Related Paper:**

 - H. Tang, T. Zhang, L. Wang, X. Ding, M. Yuan, and X. Niu, “PA-LVIO: Real-Time LiDAR-Visual-Inertial Odometry and Mapping with Pose-Only Bundle Adjustment,” Mar. 17, 2026, *arXiv*: arXiv:2603.16228. doi: [10.48550/arXiv.2603.16228](https://doi.org/10.48550/arXiv.2603.16228).
 - H. Tang, T. Zhang, L. Wang, M. Yuan, and X. Niu, “BA-LINS: A Frame-to-Frame Bundle Adjustment for LiDAR-Inertial Navigation,” *IEEE Transactions on Intelligent Transportation Systems*, vol. 26, no. 5, pp. 6621–6634, May 2025, doi: [10.1109/TITS.2024.3524569](https://doi.org/10.1109/TITS.2024.3524569).
 - H. Tang, T. Zhang, L. Wang, G. Wang, and X. Niu, “PO-VINS: An Efficient and Robust Pose-Only Visual–Inertial State Estimator With LiDAR Enhancement,” *IEEE Transactions on Instrumentation and Measurement*, vol. 74, pp. 1–17, 2025, doi: [10.1109/TIM.2025.3632451](https://doi.org/10.1109/TIM.2025.3632451).

## News

- [2026-03-19] Add the preprint paper. The codes will be released soon.

## License

PA-LVIO is released under GPLv3 license, and it can only be used for academic purposes.

## Contact

Please contact **Dr. Hailiang Tang** ([thl@whu.edu.cn](mailto:thl@whu.edu.cn)) or open an issue at this repository for any technical issues.