-- ==========================================================
-- Database Schema: k3monitoring
-- Proyek: Rancang Bangun Sistem Deteksi Pelanggaran APD K3
-- ==========================================================

CREATE DATABASE IF NOT EXISTS `k3monitoring` 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

USE `k3monitoring`;

-- ----------------------------------------------------------
-- Table structure for table `users`
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS `users` (
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `nama` VARCHAR(100) NOT NULL,
  `username` VARCHAR(50) NOT NULL UNIQUE,
  `password` VARCHAR(100) NOT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Sample admin user (admin / admin123)
INSERT IGNORE INTO `users` (`id`, `nama`, `username`, `password`) 
VALUES (1, 'Administrator K3', 'admin', 'admin123');

-- ----------------------------------------------------------
-- Table structure for table `history_pelanggaran`
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS `history_pelanggaran` (
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `jenis_pelanggaran` VARCHAR(100) NOT NULL,
  `keterangan` VARCHAR(255) NOT NULL,
  `bukti` VARCHAR(100) NOT NULL,
  `waktu` VARCHAR(50) NOT NULL,
  `akurasi` VARCHAR(20) NOT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Sample violation record for preview
INSERT IGNORE INTO `history_pelanggaran` (`id`, `jenis_pelanggaran`, `keterangan`, `bukti`, `waktu`, `akurasi`) 
VALUES 
(1, 'HELM', 'Terdeteksi TANPA HELM', 'cap_20260209_143010.jpg', '2026-02-09 14:30:10', '92.4%'),
(2, 'SEPATU', 'Terdeteksi TANPA SEPATU', 'cap_20260209_143023.jpg', '2026-02-09 14:30:23', '88.7%');
