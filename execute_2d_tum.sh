python coslam2d2d_final.py --config './configs/Tum/fr1_desk.yaml'
mv /home/shham/Co-SLAM/output/TUM/fr_desk/demo /home/shham/Co-SLAM/output/TUM/fr_desk/demo_10 && echo "Folder renamed successfully." || echo "Folder not found."

python coslam2d2d_final.py --config './configs/Tum/fr2_xyz.yaml'
mv /home/shham/Co-SLAM/output/TUM/fr_xyz/demo /home/shham/Co-SLAM/output/TUM/fr_xyz/demo_10 && echo "Folder renamed successfully." || echo "Folder not found."

python coslam2d2d_final.py --config './configs/Tum/fr3_office.yaml'
mv /home/shham/Co-SLAM/output/TUM/fr_office/demo /home/shham/Co-SLAM/output/TUM/fr_office/demo_10 && echo "Folder renamed successfully." || echo "Folder not found."
