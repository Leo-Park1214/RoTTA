import tarfile
import urllib.request
from pathlib import Path
from typing import List

from tqdm import tqdm
from torchvision.datasets import ImageFolder

from .base_dataset import TTADatasetBase, DatumList


IMAGENET_C_ARCHIVES = {
    "noise": "https://zenodo.org/records/2235448/files/noise.tar?download=1",
    "blur": "https://zenodo.org/records/2235448/files/blur.tar?download=1",
    "weather": "https://zenodo.org/records/2235448/files/weather.tar?download=1",
    "digital": "https://zenodo.org/records/2235448/files/digital.tar?download=1",
    "extra": "https://zenodo.org/records/2235448/files/extra.tar?download=1",
}

CORRUPTION_TO_ARCHIVE = {
    # noise
    "gaussian_noise": "noise",
    "shot_noise": "noise",
    "impulse_noise": "noise",

    # blur
    "defocus_blur": "blur",
    "glass_blur": "blur",
    "motion_blur": "blur",
    "zoom_blur": "blur",

    # weather
    "snow": "weather",
    "frost": "weather",
    "fog": "weather",
    "brightness": "weather",

    # digital
    "contrast": "digital",
    "elastic_transform": "digital",
    "pixelate": "digital",
    "jpeg_compression": "digital",

    # extra
    "speckle_noise": "extra",
    "gaussian_blur": "extra",
    "spatter": "extra",
    "saturate": "extra",
}


class CorruptionImageNetC(TTADatasetBase):
    """
    ImageNet-C 전용 dataset

    특징:
    - DatumList 사용: 이미지 자체를 미리 읽지 않고 경로만 저장
    - 실제 이미지 로딩/resize/totensor/normalize는 TTADatasetBase.__getitem__에서 수행
    - 필요한 corruption archive만 자동 다운로드
    - 압축 해제 성공 후 tar 파일 삭제

    Required cfg fields:
        cfg.CORRUPTION.NUM_EX
        cfg.DATA_DIR

    Optional cfg fields:
        cfg.CORRUPTION.AUTO_DOWNLOAD_IMAGENET_C (default: True)
    """

    def __init__(self, cfg, all_corruption, all_severity):
        all_corruption = [all_corruption] if not isinstance(all_corruption, list) else all_corruption
        all_severity = [all_severity] if not isinstance(all_severity, list) else all_severity

        self.corruptions = all_corruption
        self.severity = all_severity
        self.domain_id_to_name = {}

        data_source = self._build_imagenet_c(cfg)
        super().__init__(cfg, data_source)

    def _build_imagenet_c(self, cfg):
        data_root = Path(cfg.DATA_DIR)
        imagenet_c_root = data_root / "ImageNet-C"

        auto_download = getattr(cfg.CORRUPTION, "AUTO_DOWNLOAD_IMAGENET_C", True)
        if auto_download:
            self._ensure_imagenet_c_downloaded(imagenet_c_root, self.corruptions)

        data_source = []
        num_ex = int(cfg.CORRUPTION.NUM_EX)

        for i_s, severity in enumerate(self.severity):
            severity = int(severity)
            if severity < 1 or severity > 5:
                raise ValueError(f"ImageNet-C severity must be in [1, 5], got {severity}")

            for i_c, corruption in enumerate(self.corruptions):
                d_name = f"{corruption}_{severity}"
                d_id = i_s * len(self.corruptions) + i_c
                self.domain_id_to_name[d_id] = d_name

                corruption_dir = imagenet_c_root / corruption / str(severity)
                if not corruption_dir.exists():
                    raise FileNotFoundError(f"ImageNet-C path not found: {corruption_dir}")

                # transform=None 으로 두고 samples(path, label)만 사용
                ds = ImageFolder(root=str(corruption_dir), transform=None)
                samples = ds.samples

                if num_ex > 0:
                    samples = samples[:min(num_ex, len(samples))]

                for img_path, label in samples:
                    data_item = DatumList(img=str(img_path), label=int(label), domain=d_id)
                    data_source.append(data_item)

        if len(data_source) == 0:
            raise RuntimeError("No ImageNet-C samples were collected.")

        return data_source

    def _ensure_imagenet_c_downloaded(self, imagenet_c_root: Path, corruptions: List[str]):
        """
        필요한 corruption이 없으면 해당 archive만 다운로드하고,
        압축 해제 성공 후 tar 파일은 삭제.
        """
        imagenet_c_root.mkdir(parents=True, exist_ok=True)

        needed_archives = set()

        for corruption in corruptions:
            if corruption not in CORRUPTION_TO_ARCHIVE:
                raise ValueError(
                    f"Unknown ImageNet-C corruption: {corruption}. "
                    f"Valid corruptions: {sorted(CORRUPTION_TO_ARCHIVE.keys())}"
                )

            # 이미 해당 corruption 폴더가 있으면 skip
            if (imagenet_c_root / corruption).exists():
                continue

            needed_archives.add(CORRUPTION_TO_ARCHIVE[corruption])

        for archive_name in needed_archives:
            url = IMAGENET_C_ARCHIVES[archive_name]
            tar_path = imagenet_c_root / f"{archive_name}.tar"

            if not tar_path.exists():
                print(f"[ImageNet-C] Downloading {archive_name}.tar from {url}")
                self._download_with_tqdm(url, tar_path)

            try:
                print(f"[ImageNet-C] Extracting {tar_path} ...")
                with tarfile.open(tar_path, "r") as tar:
                    self._safe_extract(tar, imagenet_c_root)
            except Exception as e:
                print(f"[ImageNet-C] Extraction failed for {tar_path}: {e}")
                raise
            else:
                if tar_path.exists():
                    tar_path.unlink()
                    print(f"[ImageNet-C] Deleted archive: {tar_path}")

    def _download_with_tqdm(self, url: str, output_path: Path):
        output_path = Path(output_path)

        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=output_path.name,
        ) as pbar:

            def reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    pbar.total = total_size
                downloaded = block_num * block_size
                pbar.update(downloaded - pbar.n)

            urllib.request.urlretrieve(url, str(output_path), reporthook=reporthook)

    def _safe_extract(self, tar: tarfile.TarFile, path: Path):
        path = path.resolve()

        for member in tar.getmembers():
            member_path = (path / member.name).resolve()
            if not str(member_path).startswith(str(path)):
                raise RuntimeError(f"Unsafe path detected in tar file: {member.name}")

        tar.extractall(path=str(path))