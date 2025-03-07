/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace {
std::string GetReadImageFromDataType(DataType data_type) {
  if (data_type == DataType::FLOAT32) {
    return "read_imagef";
  } else if (data_type == DataType::FLOAT16) {
    return "read_imageh";
  } else if (data_type == DataType::INT8 || data_type == DataType::INT16 ||
             data_type == DataType::INT32) {
    return "read_imagei";
  } else if (data_type == DataType::UINT8 || data_type == DataType::UINT16 ||
             data_type == DataType::UINT32) {
    return "read_imageui";
  } else {
    return "error";
  }
}

std::string GetWriteImageFromDataType(DataType data_type) {
  if (data_type == DataType::FLOAT32) {
    return "write_imagef";
  } else if (data_type == DataType::FLOAT16) {
    return "write_imageh";
  } else if (data_type == DataType::INT8 || data_type == DataType::INT16 ||
             data_type == DataType::INT32) {
    return "write_imagei";
  } else if (data_type == DataType::UINT8 || data_type == DataType::UINT16 ||
             data_type == DataType::UINT32) {
    return "write_imageui";
  } else {
    return "error";
  }
}

std::string AddressModeToCLSampler(AddressMode address_mode) {
  switch (address_mode) {
    case AddressMode::kDontCare:
      return "smp_none";
    case AddressMode::kZero:
      return "smp_zero";
  }
}

}  // namespace

std::string ToString(TensorStorageType type) {
  switch (type) {
    case TensorStorageType::UNKNOWN:
      return "TensorStorageType::UNKNOWN";
    case TensorStorageType::BUFFER:
      return "TensorStorageType::BUFFER";
    case TensorStorageType::TEXTURE_ARRAY:
      return "TensorStorageType::TEXTURE_ARRAY";
    case TensorStorageType::TEXTURE_2D:
      return "TensorStorageType::TEXTURE_2D";
    case TensorStorageType::TEXTURE_3D:
      return "TensorStorageType::TEXTURE_3D";
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return "TensorStorageType::SINGLE_TEXTURE_2D";
    case TensorStorageType::IMAGE_BUFFER:
      return "TensorStorageType::IMAGE_BUFFER";
  }
}

TensorDescriptor::TensorDescriptor(TensorDescriptor&& desc)
    : GPUObjectDescriptor(std::move(desc)),
      data_type(desc.data_type),
      storage_type(desc.storage_type),
      layout(desc.layout),
      shape(desc.shape),
      data(std::move(desc.data)),
      use_buffer_for_write_only_2d_texture(
          desc.use_buffer_for_write_only_2d_texture),
      use_buffer_for_write_only_image_buffer(
          desc.use_buffer_for_write_only_image_buffer) {}
TensorDescriptor& TensorDescriptor::operator=(TensorDescriptor&& desc) {
  if (this != &desc) {
    std::swap(data_type, desc.data_type);
    std::swap(storage_type, desc.storage_type);
    std::swap(layout, desc.layout);
    std::swap(shape, desc.shape);
    data = std::move(desc.data);
    std::swap(use_buffer_for_write_only_2d_texture,
              desc.use_buffer_for_write_only_2d_texture);
    std::swap(use_buffer_for_write_only_image_buffer,
              desc.use_buffer_for_write_only_image_buffer);
    GPUObjectDescriptor::operator=(std::move(desc));
  }
  return *this;
}

GPUResources TensorDescriptor::GetGPUResources(const GpuInfo& gpu_info) const {
  GPUResources resources;
  resources.ints.push_back("slice_stride");
  if (HasAxis(Axis::WIDTH)) {
    resources.ints.push_back("width");
  }
  if (HasAxis(Axis::HEIGHT)) {
    resources.ints.push_back("height");
  }
  if (HasAxis(Axis::CHANNELS)) {
    resources.ints.push_back("slices");
    resources.ints.push_back("channels");
  }
  if (HasAxis(Axis::BATCH)) {
    resources.ints.push_back("batch");
  }
  if (HasAxis(Axis::DEPTH)) {
    resources.ints.push_back("depth");
  }
  if (storage_type == TensorStorageType::BUFFER) {
    GPUBufferDescriptor desc;
    desc.data_type = data_type;
    desc.access_type = access_type_;
    desc.element_size = 4;
    auto it1 = state_vars_.find("ElementsX2");
    if (it1 != state_vars_.end() && it1->second == "true") {
      desc.element_size = 8;
    }
    auto it2 = state_vars_.find("ElementsX4");
    if (it2 != state_vars_.end() && it2->second == "true") {
      desc.element_size = 16;
    }
    resources.buffers.push_back({"buffer", desc});
  } else if (storage_type == TensorStorageType::SINGLE_TEXTURE_2D ||
             storage_type == TensorStorageType::TEXTURE_2D) {
    if (access_type_ == AccessType::WRITE &&
        use_buffer_for_write_only_2d_texture) {
      resources.ints.push_back("aligned_texture_width");
      GPUBufferDescriptor desc;
      desc.data_type = data_type;
      desc.access_type = access_type_;
      desc.element_size = 4;
      resources.buffers.push_back({"buffer", desc});
    } else {
      GPUImage2DDescriptor desc;
      desc.data_type = data_type;
      desc.normalized = false;
      desc.access_type = access_type_;
      resources.images2d.push_back({"image2d", desc});
    }
  } else if (storage_type == TensorStorageType::TEXTURE_ARRAY) {
    GPUImage2DArrayDescriptor desc;
    desc.data_type = data_type;
    desc.access_type = access_type_;
    resources.image2d_arrays.push_back({"image2d_array", desc});
  } else if (storage_type == TensorStorageType::TEXTURE_3D) {
    GPUImage3DDescriptor desc;
    desc.data_type = data_type;
    desc.access_type = access_type_;
    resources.images3d.push_back({"image3d", desc});
  } else if (storage_type == TensorStorageType::IMAGE_BUFFER) {
    if (access_type_ == AccessType::WRITE &&
        use_buffer_for_write_only_image_buffer) {
      GPUBufferDescriptor desc;
      desc.data_type = data_type;
      desc.access_type = access_type_;
      desc.element_size = 4;
      resources.buffers.push_back({"buffer", desc});
    } else {
      GPUImageBufferDescriptor desc;
      desc.data_type = data_type;
      desc.access_type = access_type_;
      resources.image_buffers.push_back({"image_buffer", desc});
    }
  }
  return resources;
}

absl::Status TensorDescriptor::PerformSelector(
    const GpuInfo& gpu_info, const std::string& selector,
    const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (selector == "Width") {
    *result = "width";
    return absl::OkStatus();
  } else if (selector == "Height") {
    *result = "height";
    return absl::OkStatus();
  } else if (selector == "Slices") {
    *result = "slices";
    return absl::OkStatus();
  } else if (selector == "SliceStride") {
    *result = "slice_stride";
    return absl::OkStatus();
  } else if (selector == "Channels") {
    *result = "channels";
    return absl::OkStatus();
  } else if (selector == "Batch") {
    if (HasAxis(Axis::BATCH)) {
      *result = "batch";
    } else {
      *result = "1";
    }
    return absl::OkStatus();
  } else if (selector == "Depth") {
    *result = "depth";
    return absl::OkStatus();
  } else if (selector == "SetBatchRef") {
    if (args.size() != 1) {
      return absl::InvalidArgumentError(
          "Unsupported arguments in SetBatchRef selector");
    }
    state_vars_["batch_id"] = args[0];
    *result = "";
    return absl::OkStatus();
  } else if (selector == "Read") {
    return PerformReadSelector(gpu_info, args, template_args, result);
  } else if (selector == "ReadNearest") {
    return PerformReadNearestSelector(gpu_info, args, result);
  } else if (selector == "ReadBilinear") {
    return PerformReadBilinearSelector(gpu_info, args, result);
  } else if (selector == "Write") {
    return PerformWriteSelector(gpu_info, args, result);
  } else if (selector == "WriteLinear") {
    return PerformWriteLinearSelector(gpu_info, args, result);
  } else if (selector == "Write2D") {
    return PerformWrite2DSelector(gpu_info, args, result);
  } else if (selector == "GetAddress") {
    return PerformGetAddressSelector(args, result);
  } else if (selector == "GetPtrWithSliceOffset") {
    return PerformGetPtrWithSliceOffsetSelector(args, result);
  } else if (selector == "GetWHOffset") {
    return PerformGetWHOffsetSelector(args, result);
  } else if (selector == "GetHandle") {
    return PerformGetHandleSelector(args, result);
  } else {
    return absl::NotFoundError(absl::StrCat(
        "TensorDescriptor don't have selector with name - ", selector));
  }
}

absl::Status TensorDescriptor::PerformReadSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  DataType read_as_type = data_type;
  if (!template_args.empty()) {
    if (template_args.size() != 1) {
      return absl::NotFoundError(
          "Unrecognized Read selector template arguments.");
    } else {
      RETURN_IF_ERROR(
          GetDataTypeFromTemplateArgs(template_args[0], &read_as_type));
    }
  }
  if (args.size() == 1) {  // function overload for 1D linear types.
    if (storage_type == TensorStorageType::BUFFER ||
        storage_type == TensorStorageType::IMAGE_BUFFER) {
      *result = Read(gpu_info, read_as_type, {args[0]});
      return absl::OkStatus();
    } else {
      return absl::InvalidArgumentError(
          "Read selector with single argument can be used only with linear "
          "storage types(BUFFER or IMAGE_BUFFER)");
    }
  }
  std::string xc;
  std::string yc;
  std::string zc;
  std::string sc;
  std::string bc;
  bool parsed = ParseCoordsFromArgs(args, 0, &xc, &yc, &zc, &sc, &bc);
  if (args.size() < 2 || !parsed) {
    return absl::NotFoundError("Unrecognized Read selector");
  }

  *result = Read(gpu_info, read_as_type, GetPhysicalCoords(xc, yc, zc, sc, bc));
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformReadNearestSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  if (IsBatchedWidth()) {
    return absl::NotFoundError(
        "ReadNearest can not be used with BatchedWidth.");
  }
  // ReadNearest(result, fc_x, fc_y, {fc_z}, slice);
  if (!((args.size() == 5 && HasAxis(Axis::DEPTH)) || args.size() == 4)) {
    return absl::NotFoundError("Unrecognized ReadNearest selector");
  }
  std::vector<std::string> coord_args =
      std::vector<std::string>(args.begin() + 1, args.end());
  std::string c;
  c += "  {\n";
  c += "  int coord_x_TMP = INIT_INT(" + coord_args[0] + ");\n";
  c += "  coord_x_TMP = max(coord_x_TMP, 0);\n";
  c += "  coord_x_TMP = min(coord_x_TMP, width - 1);\n";
  coord_args[0] = "coord_x_TMP";
  c += "  int coord_y_TMP = INIT_INT(" + coord_args[1] + ");\n";
  c += "  coord_y_TMP = max(coord_y_TMP, 0);\n";
  c += "  coord_y_TMP = min(coord_y_TMP, height - 1);\n";
  coord_args[1] = "coord_y_TMP";
  if (HasAxis(Axis::DEPTH)) {
    c += "  int coord_z_TMP = INIT_INT(" + coord_args[2] + ");\n";
    c += "  coord_z_TMP = max(coord_z_TMP, 0);\n";
    c += "  coord_z_TMP = min(coord_z_TMP, depth - 1);\n";
    coord_args[2] = "coord_z_TMP";
  }
  std::string src_value;
  RETURN_IF_ERROR(PerformReadSelector(gpu_info, coord_args, {}, &src_value));
  c += "  " + args[0] + " = " + src_value + ";\n";
  c += "  }";
  *result = c;
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformReadBilinearSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  if (IsBatchedWidth()) {
    return absl::NotFoundError(
        "ReadBilinear can not be used with BatchedWidth.");
  }
  // ReadBilinear(result, fc_x, fc_y, {fc_z}, slice);
  if (!((args.size() == 5 && HasAxis(Axis::DEPTH)) || args.size() == 4)) {
    return absl::NotFoundError("Unrecognized ReadBilinear selector");
  }
  std::vector<std::string> coord_args =
      std::vector<std::string>(args.begin() + 1, args.end());
  std::string c;
  c += "  {\n";
  c += "  float f_x_TMP = floor(" + coord_args[0] + ");\n";
  c += "  float x_scale_TMP = (" + coord_args[0] + ") - f_x_TMP;\n";
  c += "  int i_x_TMP = INIT_INT(f_x_TMP);\n";
  c += "  int start_x_TMP = max(i_x_TMP, 0);\n";
  c += "  int end_x_TMP = min(i_x_TMP + 1, width - 1);\n";
  c += "  float f_y_TMP = floor(" + coord_args[1] + ");\n";
  c += "  float y_scale_TMP = (" + coord_args[1] + ") - f_y_TMP;\n";
  c += "  int i_y_TMP = INIT_INT(f_y_TMP);\n";
  c += "  int start_y_TMP = max(i_y_TMP, 0);\n";
  c += "  int end_y_TMP = min(i_y_TMP + 1, height - 1);\n";
  if (HasAxis(Axis::DEPTH)) {
    // 3d bilinear read, x, y, z
    c += "  float f_z_TMP = floor(" + coord_args[2] + ");\n";
    c += "  float z_scale_TMP = (" + coord_args[2] + ") - f_z_TMP;\n";
    c += "  int i_z_TMP = INIT_INT(f_z_TMP);\n";
    c += "  int start_z_TMP = max(i_z_TMP, 0);\n";
    c += "  int end_z_TMP = min(i_z_TMP + 1, depth - 1);\n";
    int index = 0;
    for (const auto& src_z : {"start_z_TMP", "end_z_TMP"}) {
      for (const auto& src_y : {"start_y_TMP", "end_y_TMP"}) {
        for (const auto& src_x : {"start_x_TMP", "end_x_TMP"}) {
          coord_args[0] = src_x;
          coord_args[1] = src_y;
          coord_args[2] = src_z;
          std::string src_value;
          RETURN_IF_ERROR(
              PerformReadSelector(gpu_info, coord_args, {"float"}, &src_value));
          c += "  float4 src" + std::to_string(index) + "_TMP = " + src_value +
               ";\n";
          index++;
        }
      }
    }
    c += "  float4 t0_TMP = mix(mix(src0_TMP, src1_TMP, x_scale_TMP), "
         "mix(src2_TMP, src3_TMP, x_scale_TMP), y_scale_TMP);\n";
    c += "  float4 t1_TMP = mix(mix(src4_TMP, src5_TMP, x_scale_TMP), "
         "mix(src6_TMP, src7_TMP, x_scale_TMP), y_scale_TMP);\n";
    c += "  " + args[0] + " = TO_FLT4(mix(t0_TMP, t1_TMP, z_scale_TMP));\n";
  } else {
    // 2d bilinear read, x, y
    int index = 0;
    for (const auto& src_y : {"start_y_TMP", "end_y_TMP"}) {
      for (const auto& src_x : {"start_x_TMP", "end_x_TMP"}) {
        coord_args[0] = src_x;
        coord_args[1] = src_y;
        std::string src_value;
        RETURN_IF_ERROR(
            PerformReadSelector(gpu_info, coord_args, {"float"}, &src_value));
        c += "  float4 src" + std::to_string(index) + "_TMP = " + src_value +
             ";\n";
        index++;
      }
    }
    c += "  " + args[0] +
         " = TO_FLT4(mix(mix(src0_TMP, src1_TMP, x_scale_TMP), mix(src2_TMP, "
         "src3_TMP, x_scale_TMP), y_scale_TMP));\n";
  }
  c += "  }";
  *result = c;
  return absl::OkStatus();
}

absl::Status TensorDescriptor::GetLinkingContextFromWriteSelector(
    const std::vector<std::string>& args, std::string* value_name,
    std::string* x_coord, std::string* y_coord, std::string* s_coord) const {
  std::string xc;
  std::string yc;
  std::string zc;
  std::string sc;
  std::string bc;
  bool parsed = ParseCoordsFromArgs(args, 1, &xc, &yc, &zc, &sc, &bc);
  if (args.size() < 2 || !parsed) {
    return absl::NotFoundError("Unrecognized Write selector");
  }
  *value_name = args[0];
  if (HasAxis(Axis::BATCH) && !IsBatchedWidth()) {
    *x_coord = absl::StrCat("((", xc, ") * batch + (", bc, "))");
  } else {
    *x_coord = absl::StrCat("(", xc, ")");
  }
  *y_coord = absl::StrCat("(", yc, ")");
  *s_coord = absl::StrCat("(", sc, ")");
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformWriteSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  std::string xc;
  std::string yc;
  std::string zc;
  std::string sc;
  std::string bc;
  bool parsed = ParseCoordsFromArgs(args, 1, &xc, &yc, &zc, &sc, &bc);
  if (args.size() < 2 || !parsed) {
    return absl::NotFoundError("Unrecognized Write selector");
  }
  *result = Write(gpu_info, args[0], GetPhysicalCoords(xc, yc, zc, sc, bc));
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformWriteLinearSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  if (storage_type != TensorStorageType::BUFFER &&
      storage_type != TensorStorageType::IMAGE_BUFFER) {
    return absl::InvalidArgumentError(
        "WriteLinear selector can be used only with linear "
        "storages(BUFFER/IMAGE_BUFFER)");
  }
  if (args.size() != 2) {
    return absl::NotFoundError("Unrecognized WriteLinear selector");
  }
  *result = Write(gpu_info, args[0], {args[1]});
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformWrite2DSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  if (storage_type != TensorStorageType::TEXTURE_2D) {
    return absl::InvalidArgumentError(
        "Write2D selector can be used only with 2d "
        "storages(TEXTURE_2D)");
  }
  if (args.size() != 3) {
    return absl::NotFoundError("Unrecognized Write2D selector");
  }
  *result = Write(gpu_info, args[0], {args[1], args[2]});
  return absl::OkStatus();
}

std::string TensorDescriptor::Read(
    const GpuInfo& gpu_info, DataType read_as_type,
    const std::vector<std::string>& coords) const {
  const bool need_conversion = read_as_type != data_type;
  const std::string metal_type =
      read_as_type == DataType::FLOAT32 ? "float4" : "half4";
  switch (storage_type) {
    case TensorStorageType::BUFFER:
      if (gpu_info.IsGlsl()) {
        if (data_type == DataType::FLOAT16 &&
            !gpu_info.IsGlslSupportsExplicitFp16()) {
          return absl::StrCat("vec4(unpackHalf2x16(buffer[", coords[0],
                              "].x), unpackHalf2x16(buffer[", coords[0],
                              "].y))");
        } else {
          return absl::StrCat("buffer[", coords[0], "]");
        }
      }
      if (read_as_type == data_type) {
        return absl::StrCat("buffer[", coords[0], "]");
      } else {
        std::string conversion;
        if (gpu_info.IsApiMetal()) {
          conversion = metal_type;
        } else if (gpu_info.IsApiOpenCl()) {
          if (read_as_type == DataType::FLOAT16) {
            conversion = "convert_half4";
          } else if (read_as_type == DataType::FLOAT32) {
            conversion = "convert_float4";
          }
        }
        return absl::StrCat(conversion, "(buffer[", coords[0], "])");
      }
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      if (gpu_info.IsApiOpenCl()) {
        return absl::Substitute("$0(image2d, $1, (int2)($2, $3))",
                                GetReadImageFromDataType(read_as_type),
                                AddressModeToCLSampler(AddressModeFromState()),
                                coords[0], coords[1]);
      } else if (gpu_info.IsApiMetal()) {
        std::string result = absl::Substitute("image2d.read(ushort2($0, $1))",
                                              coords[0], coords[1]);
        if (need_conversion) {
          result = metal_type + "(" + result + ")";
        }
        return result;
      } else if (gpu_info.IsGlsl()) {
        std::string result = "texelFetch(image2d, ivec2(" + coords[0] + ", " +
                             coords[1] + "), 0)";
        if (data_type == DataType::FLOAT16 &&
            gpu_info.IsGlslSupportsExplicitFp16()) {
          result = "f16vec4(" + result + ")";
        }
        return result;
      } else {
        return "";
      }
    case TensorStorageType::TEXTURE_3D:
      if (gpu_info.IsApiOpenCl()) {
        return absl::Substitute("$0(image3d, $1, (int4)($2, $3, $4, 0))",
                                GetReadImageFromDataType(read_as_type),
                                AddressModeToCLSampler(AddressModeFromState()),
                                coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsApiMetal()) {
        std::string result =
            absl::Substitute("image3d.read(ushort3($0, $1, $2))", coords[0],
                             coords[1], coords[2]);
        if (need_conversion) {
          result = metal_type + "(" + result + ")";
        }
        return result;
      } else if (gpu_info.IsGlsl()) {
        std::string result = "texelFetch(image3d, ivec3(" + coords[0] + ", " +
                             coords[1] + ", " + coords[2] + "), 0)";
        if (data_type == DataType::FLOAT16 &&
            gpu_info.IsGlslSupportsExplicitFp16()) {
          result = "f16vec4(" + result + ")";
        }
        return result;
      } else {
        return "";
      }
    case TensorStorageType::TEXTURE_ARRAY:
      if (gpu_info.IsApiOpenCl()) {
        return absl::Substitute("$0(image2d_array, $1, (int4)($2, $3, $4, 0))",
                                GetReadImageFromDataType(read_as_type),
                                AddressModeToCLSampler(AddressModeFromState()),
                                coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsApiMetal()) {
        std::string result =
            absl::Substitute("image2d_array.read(ushort2($0, $1), $2)",
                             coords[0], coords[1], coords[2]);
        if (need_conversion) {
          result = metal_type + "(" + result + ")";
        }
        return result;
      } else if (gpu_info.IsGlsl()) {
        std::string result = "texelFetch(image2d_array, ivec3(" + coords[0] +
                             ", " + coords[1] + ", " + coords[2] + "), 0)";
        if (data_type == DataType::FLOAT16 &&
            gpu_info.IsGlslSupportsExplicitFp16()) {
          result = "f16vec4(" + result + ")";
        }
        return result;
      } else {
        return "";
      }
    case TensorStorageType::IMAGE_BUFFER:
      if (gpu_info.IsApiOpenCl()) {
        return absl::StrCat(GetReadImageFromDataType(read_as_type),
                            "(image_buffer, ", coords[0], ")");
      } else if (gpu_info.IsApiMetal()) {
        std::string result =
            absl::Substitute("image_buffer.read(uint($0))", coords[0]);
        if (need_conversion) {
          result = metal_type + "(" + result + ")";
        }
        return result;
      } else if (gpu_info.IsGlsl()) {
        std::string result = "texelFetch(image_buffer, " + coords[0] + ")";
        if (data_type == DataType::FLOAT16 &&
            gpu_info.IsGlslSupportsExplicitFp16()) {
          result = "f16vec4(" + result + ")";
        }
        return result;
      } else {
        return "";
      }
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::string TensorDescriptor::Write(
    const GpuInfo& gpu_info, const std::string& var_name,
    const std::vector<std::string>& coords) const {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      if (gpu_info.IsApiOpenCl()) {
        if (use_buffer_for_write_only_image_buffer) {
          return absl::StrCat("buffer[", coords[0], "] = ", var_name);
        } else {
          return absl::Substitute("$0(image_buffer, $1, $2)",
                                  GetWriteImageFromDataType(data_type),
                                  coords[0], var_name);
        }
      } else if (gpu_info.IsApiMetal()) {
        if (use_buffer_for_write_only_image_buffer) {
          return absl::StrCat("buffer[", coords[0], "] = ", var_name);
        } else {
          return absl::Substitute("image_buffer.write($0, uint($1))", var_name,
                                  coords[0]);
        }
      } else if (gpu_info.IsGlsl()) {
        if (data_type == DataType::FLOAT16 &&
            !gpu_info.IsGlslSupportsExplicitFp16()) {
          return absl::StrCat("buffer[", coords[0], "] = uvec2(packHalf2x16(",
                              var_name, ".xy), packHalf2x16(", var_name,
                              ".zw))");
        } else {
          return absl::StrCat("buffer[", coords[0], "] = ", var_name);
        }
      } else {
        return absl::StrCat("buffer[", coords[0], "] = ", var_name);
      }
    case TensorStorageType::SINGLE_TEXTURE_2D:
    case TensorStorageType::TEXTURE_2D:
      if (gpu_info.IsApiOpenCl()) {
        if (use_buffer_for_write_only_2d_texture) {
          return absl::Substitute(
              "buffer[($2) * aligned_texture_width + ($1)] = $0", var_name,
              coords[0], coords[1]);
        } else {
          return absl::Substitute("$0(image2d, (int2)($1, $2), $3)",
                                  GetWriteImageFromDataType(data_type),
                                  coords[0], coords[1], var_name);
        }
      } else if (gpu_info.IsApiMetal()) {
        if (use_buffer_for_write_only_2d_texture) {
          return absl::Substitute(
              "buffer[($2) * aligned_texture_width + ($1)] = $0", var_name,
              coords[0], coords[1]);
        } else {
          return absl::Substitute("image2d.write($0, ushort2($1, $2))",
                                  var_name, coords[0], coords[1]);
        }
      } else if (gpu_info.IsGlsl()) {
        return absl::Substitute("imageStore(image2d, ivec2($0, $1), $2)",
                                coords[0], coords[1], var_name);
      } else {
        return "";
      }
    case TensorStorageType::TEXTURE_3D:
      if (gpu_info.IsApiOpenCl()) {
        return absl::Substitute("$0(image3d, (int4)($1, $2, $3, 0), $4)",
                                GetWriteImageFromDataType(data_type), coords[0],
                                coords[1], coords[2], var_name);
      } else if (gpu_info.IsApiMetal()) {
        return absl::Substitute("image3d.write($0, ushort3($1, $2, $3))",
                                var_name, coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsGlsl()) {
        return absl::Substitute("imageStore(image3d, ivec3($0, $1, $2), $3)",
                                coords[0], coords[1], coords[2], var_name);
      } else {
        return "";
      }
    case TensorStorageType::TEXTURE_ARRAY:
      if (gpu_info.IsApiOpenCl()) {
        return absl::Substitute("$0(image2d_array, (int4)($1, $2, $3, 0), $4)",
                                GetWriteImageFromDataType(data_type), coords[0],
                                coords[1], coords[2], var_name);
      } else if (gpu_info.IsApiMetal()) {
        return absl::Substitute("image2d_array.write($0, ushort2($1, $2), $3)",
                                var_name, coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsGlsl()) {
        return absl::Substitute(
            "imageStore(image2d_array, ivec3($0, $1, $2), $3)", coords[0],
            coords[1], coords[2], var_name);
      } else {
        return "";
      }
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

absl::Status TensorDescriptor::PerformGetAddressSelector(
    const std::vector<std::string>& args, std::string* result) const {
  std::string xc;
  std::string yc;
  std::string zc;
  std::string sc;
  std::string bc;
  bool parsed = ParseCoordsFromArgs(args, 1, &xc, &yc, &zc, &sc, &bc);
  if (args.size() < 3 || !parsed) {
    return absl::NotFoundError("Unrecognized GetAddress selector");
  }

  *result = DeclareAddress(args[0],
                           GetGlobalAddressNoDeclaration(xc, yc, zc, sc, bc));
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformGetPtrWithSliceOffsetSelector(
    const std::vector<std::string>& args, std::string* result) const {
  if (storage_type != TensorStorageType::BUFFER) {
    return absl::InvalidArgumentError(
        "GetPtrWithSliceOffset selector can be used only with BUFFER");
  }
  if (args.size() != 1) {
    return absl::NotFoundError(absl::StrCat(
        "GetPtrWithSliceOffset require one argument(slice coordinate), but ",
        args.size(), " was passed"));
  }
  *result = absl::StrCat("buffer + ", args[0], " * slice_stride");
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformGetWHOffsetSelector(
    const std::vector<std::string>& args, std::string* result) const {
  if (storage_type != TensorStorageType::BUFFER &&
      storage_type != TensorStorageType::IMAGE_BUFFER) {
    return absl::InvalidArgumentError(
        "GetWHOffset selector can be used only with BUFFER/IMAGE_BUFFER");
  }
  if (args.size() != 2) {
    return absl::NotFoundError(absl::StrCat(
        "GetWHOffset require two arguments(X and Y coordinates), but ",
        args.size(), " was passed"));
  }
  if (HasAxis(Axis::BATCH) && !IsBatchedWidth()) {
    auto it = state_vars_.find("batch_id");
    std::string batch_id;
    if (it == state_vars_.end()) {
      return absl::NotFoundError(
          "Not found batch_id. Should be setted up by SetBatchRef(). method");
    } else {
      batch_id = it->second;
    }
    *result = absl::StrCat("((", args[1], ") * width + (", args[0],
                           ")) * batch + (", batch_id, ")");
  } else {
    *result = absl::StrCat("(", args[1], ") * width + (", args[0], ")");
  }
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformGetHandleSelector(
    const std::vector<std::string>& args, std::string* result) const {
  if (!args.empty()) {
    return absl::NotFoundError(
        absl::StrCat("GetHandle does not require arguments, but ", args.size(),
                     " was passed"));
  }
  switch (storage_type) {
    case TensorStorageType::BUFFER:
      *result = "buffer";
      return absl::OkStatus();
    case TensorStorageType::IMAGE_BUFFER:
      if (access_type_ == AccessType::READ) {
        *result = "image_buffer";
      } else {
        *result = "buffer";
      }
      return absl::OkStatus();
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      *result = "image2d";
      return absl::OkStatus();
    case TensorStorageType::TEXTURE_ARRAY:
      *result = "image2d_array";
      return absl::OkStatus();
    case TensorStorageType::TEXTURE_3D:
      *result = "image3d";
      return absl::OkStatus();
    case TensorStorageType::UNKNOWN:
      return absl::UnavailableError("Unknown type");
  }
}

std::string TensorDescriptor::DeclareAddress(const std::string& var_name,
                                             const std::string& address) const {
  return absl::StrCat(StorageTypeToAddressType(), " ", var_name, " = ", address,
                      ";");
}

std::string TensorDescriptor::StorageTypeToAddressType() const {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return "int";
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return "int2";
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return "int4";
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoordsWHS(
    const std::string& x, const std::string& y, const std::string& s) const {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return {
          absl::Substitute("((($2) * height + ($1)) * width + ($0))", x, y, s)};
    case TensorStorageType::TEXTURE_2D:
      return {absl::Substitute("($0)", x),
              absl::Substitute("(($0) * slices + ($1))", y, s)};
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {absl::Substitute("($0)", x), absl::Substitute("($0)", y)};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {absl::Substitute("($0)", x), absl::Substitute("($0)", y),
              absl::Substitute("($0)", s)};
    case TensorStorageType::UNKNOWN:
      return {""};
    default:
      return {""};
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoordsWHSB(
    const std::string& x, const std::string& y, const std::string& s,
    const std::string& b) const {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return {absl::Substitute(
          "(((($3) * height + $2) * width + ($1)) * batch + ($0))", b, x, y,
          s)};
    case TensorStorageType::TEXTURE_2D:
      return {absl::Substitute("(($0) * batch + ($1))", x, b),
              absl::Substitute("(($0) * slices + ($1))", y, s)};
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {absl::Substitute("(($0) * batch + ($1))", x, b),
              absl::Substitute("($0)", y)};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {absl::Substitute("(($0) * batch + ($1))", x, b),
              absl::Substitute("($0)", y), absl::Substitute("($0)", s)};
    case TensorStorageType::UNKNOWN:
      return {""};
    default:
      return {""};
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoordsWHDS(
    const std::string& x, const std::string& y, const std::string& z,
    const std::string& s) const {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return {absl::Substitute(
          "(((($3) * slices + ($2)) * height + ($1)) * width + ($0))", x, y, s,
          z)};
    case TensorStorageType::TEXTURE_2D:
      return {absl::Substitute("(($0) * depth + ($1))", x, z),
              absl::Substitute("(($0) * slices + ($1))", y, s)};
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {absl::Substitute("(($0) * depth + ($1))", x, z),
              absl::Substitute("($0)", y)};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {absl::Substitute("($0)", x), absl::Substitute("($0)", y),
              absl::Substitute("(($0) * slices + ($1))", z, s)};
    case TensorStorageType::UNKNOWN:
      return {""};
    default:
      return {""};
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoordsWHDSB(
    const std::string& x, const std::string& y, const std::string& z,
    const std::string& s, const std::string& b) const {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return {absl::Substitute(
          "((((($4) * slices + ($3)) * height + $2) * width + ($1)) * batch + "
          "($0))",
          b, x, y, s, z)};
    case TensorStorageType::TEXTURE_2D:
      return {absl::Substitute("((($0)*batch + ($1))*depth + ($2))", x, b, z),
              absl::Substitute("(($0) * slices + ($1))", y, s)};
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {absl::Substitute("((($0)*batch + ($1))*depth + ($2))", x, b, z),
              absl::Substitute("($0)", y)};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {absl::Substitute("(($0) * batch + ($1))", x, b),
              absl::Substitute("($0)", y),
              absl::Substitute("(($0) * slices + ($1))", z, s)};
    case TensorStorageType::UNKNOWN:
      return {""};
    default:
      return {""};
  }
}

std::string TensorDescriptor::GetGlobalAddressNoDeclaration(
    const std::string& xc, const std::string& yc, const std::string& zc,
    const std::string& sc, const std::string& bc) const {
  auto coords = GetPhysicalCoords(xc, yc, zc, sc, bc);
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER: {
      return coords[0];
    }
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::Substitute("(int2)($0, $1)", coords[0], coords[1]);
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return absl::Substitute("(int4)($0, $1, $2, 0)", coords[0], coords[1],
                              coords[2]);
    case TensorStorageType::UNKNOWN:
      return "error";
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoords(
    const std::string& xc, const std::string& yc, const std::string& zc,
    const std::string& sc, const std::string& bc) const {
  if (layout == Layout::HWC || (IsBatchedWidth() && layout == Layout::BHWC)) {
    return GetPhysicalCoordsWHS(xc, yc, sc);
  } else if (layout == Layout::BHWC) {
    return GetPhysicalCoordsWHSB(xc, yc, sc, bc);
  } else if (layout == Layout::HWDC ||
             (IsBatchedWidth() && layout == Layout::BHWDC)) {
    return GetPhysicalCoordsWHDS(xc, yc, zc, sc);
  } else if (layout == Layout::BHWDC) {
    return GetPhysicalCoordsWHDSB(xc, yc, zc, sc, bc);
  } else {
    return {""};
  }
}

absl::Status TensorDescriptor::GetDataTypeFromTemplateArgs(
    const std::string& template_arg, DataType* result) const {
  std::string read_type = template_arg;
  if (read_type == "FLT" || read_type == "ACCUM_FLT") {
    auto it = state_vars_.find(read_type);
    if (it == state_vars_.end()) {
      return absl::UnavailableError(absl::StrCat(
          "Read selector template argument ", read_type, " uninitialized."));
    } else {
      read_type = it->second;
    }
  }

  if (read_type == "half") {
    *result = DataType::FLOAT16;
  } else if (read_type == "float") {
    *result = DataType::FLOAT32;
  } else {
    return absl::NotFoundError(absl::StrCat(
        "Unrecognized Read selector template argument - ", read_type));
  }
  return absl::OkStatus();
}

bool TensorDescriptor::HasAxis(Axis axis) const {
  if (axis == Axis::WIDTH || axis == Axis::HEIGHT || axis == Axis::CHANNELS) {
    return true;
  }
  if (axis == Axis::BATCH &&
      (layout == Layout::BHWC || layout == Layout::BHWDC)) {
    return true;
  }
  if (axis == Axis::DEPTH &&
      (layout == Layout::HWDC || layout == Layout::BHWDC)) {
    return true;
  }
  return false;
}

int TensorDescriptor::GetWidthSize(BHWDC shape) const {
  int width = shape.w;
  auto it = state_vars_.find("BatchedWidth");
  if (it != state_vars_.end() && it->second == "true") {
    width *= shape.b;
  }
  auto it1 = state_vars_.find("ElementsX2");
  if (it1 != state_vars_.end() && it1->second == "true") {
    width /= 2;
  }
  auto it2 = state_vars_.find("ElementsX4");
  if (it2 != state_vars_.end() && it2->second == "true") {
    width /= 4;
  }
  return width;
}

int TensorDescriptor::GetSliceStrideSize(BHWDC shape) const {
  if (IsBatchedWidth()) {
    return GetWidthSize(shape) * shape.h;
  } else {
    if (HasAxis(Axis::BATCH)) {
      return GetWidthSize(shape) * shape.h * shape.b;
    } else {
      return GetWidthSize(shape) * shape.h;
    }
  }
}

void TensorDescriptor::SetAddressMode(AddressMode mode) {
  if (mode == AddressMode::kZero) {
    state_vars_["TextureMode"] = "ZERO";
  } else {
    state_vars_["TextureMode"] = "DONT_CARE";
  }
}

bool TensorDescriptor::ParseCoordsFromArgs(const std::vector<std::string>& args,
                                           int offset, std::string* xc,
                                           std::string* yc, std::string* zc,
                                           std::string* sc,
                                           std::string* bc) const {
  if (HasAxis(Axis::WIDTH)) {
    if (offset >= args.size()) return false;
    *xc = args[offset++];
  }
  if (HasAxis(Axis::HEIGHT)) {
    if (offset >= args.size()) return false;
    *yc = args[offset++];
  }
  if (HasAxis(Axis::DEPTH)) {
    if (offset >= args.size()) return false;
    *zc = args[offset++];
  }
  if (HasAxis(Axis::CHANNELS)) {
    if (offset >= args.size()) {
      auto it = state_vars_.find("slice_id");
      if (it == state_vars_.end()) {
        return false;
      } else {
        *sc = it->second;
      }
    } else {
      *sc = args[offset++];
    }
  }
  if (HasAxis(Axis::BATCH) && !IsBatchedWidth()) {
    if (offset >= args.size()) {
      auto it = state_vars_.find("batch_id");
      if (it == state_vars_.end()) {
        return false;
      } else {
        *bc = it->second;
      }
    } else {
      *bc = args[offset++];
    }
  }
  return true;
}

bool TensorDescriptor::IsBatchedWidth() const {
  auto it = state_vars_.find("BatchedWidth");
  return it != state_vars_.end() && it->second == "true";
}

AddressMode TensorDescriptor::AddressModeFromState() const {
  auto it = state_vars_.find("TextureMode");
  if (it != state_vars_.end()) {
    if (it->second == "ZERO") {
      return AddressMode::kZero;
    } else {
      return AddressMode::kDontCare;
    }
  } else {
    return AddressMode::kDontCare;
  }
}

void TensorDescriptor::UploadData(
    const tflite::gpu::Tensor<BHWC, DataType::FLOAT32>& src) {
  shape = BHWDC(src.shape.b, src.shape.h, src.shape.w, 1, src.shape.c);
  UploadData(src.data.data());
}

void TensorDescriptor::UploadData(
    const tflite::gpu::Tensor<BHWC, DataType::INT32>& src) {
  shape = BHWDC(src.shape.b, src.shape.h, src.shape.w, 1, src.shape.c);
  UploadData(src.data.data());
}

void TensorDescriptor::UploadData(
    const tflite::gpu::Tensor<HWC, DataType::FLOAT32>& src) {
  shape = BHWDC(1, src.shape.h, src.shape.w, 1, src.shape.c);
  UploadData(src.data.data());
}

void TensorDescriptor::UploadData(
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src) {
  shape = BHWDC(1, 1, 1, 1, src.shape.v);
  UploadData(src.data.data());
}

void TensorDescriptor::UploadData(const float* src) {
  int aligned_channels = storage_type == TensorStorageType::SINGLE_TEXTURE_2D
                             ? shape.c
                             : AlignByN(shape.c, 4);
  int elements_count = shape.b * shape.w * shape.h * shape.d * aligned_channels;
  data.resize(elements_count * SizeOf(data_type));
  if (data_type == DataType::FLOAT32) {
    float* gpu_data = reinterpret_cast<float*>(data.data());
    DataFromBHWDC(src, shape, *this, gpu_data);
  } else {
    half* gpu_data = reinterpret_cast<half*>(data.data());
    DataFromBHWDC(src, shape, *this, gpu_data);
  }
}

void TensorDescriptor::UploadData(const int32_t* src) {
  int aligned_channels = storage_type == TensorStorageType::SINGLE_TEXTURE_2D
                             ? shape.c
                             : AlignByN(shape.c, 4);
  int elements_count = shape.b * shape.w * shape.h * shape.d * aligned_channels;
  data.resize(elements_count * SizeOf(data_type));
  int32_t* gpu_data = reinterpret_cast<int32_t*>(data.data());
  DataFromBHWDC(src, shape, *this, gpu_data);
}

void TensorDescriptor::DownloadData(
    tflite::gpu::Tensor<BHWC, DataType::FLOAT32>* dst) {
  dst->shape = BHWC(shape.b, shape.h, shape.w, shape.c);
  dst->data.resize(dst->shape.DimensionsProduct(), 0.0f);
  DownloadData(dst->data.data());
}
void TensorDescriptor::DownloadData(
    tflite::gpu::Tensor<BHWC, DataType::INT32>* dst) {
  dst->shape = BHWC(shape.b, shape.h, shape.w, shape.c);
  dst->data.resize(dst->shape.DimensionsProduct(), 0);
  DownloadData(dst->data.data());
}

void TensorDescriptor::DownloadData(float* dst) {
  int aligned_channels = storage_type == TensorStorageType::SINGLE_TEXTURE_2D
                             ? shape.c
                             : AlignByN(shape.c, 4);
  int elements_count = shape.b * shape.w * shape.h * shape.d * aligned_channels;
  data.resize(elements_count * SizeOf(data_type));
  if (data_type == DataType::FLOAT32) {
    float* gpu_data = reinterpret_cast<float*>(data.data());
    DataToBHWDC(gpu_data, shape, *this, dst);
  } else {
    half* gpu_data = reinterpret_cast<half*>(data.data());
    DataToBHWDC(gpu_data, shape, *this, dst);
  }
}

void TensorDescriptor::DownloadData(int32_t* dst) {
  int aligned_channels = storage_type == TensorStorageType::SINGLE_TEXTURE_2D
                             ? shape.c
                             : AlignByN(shape.c, 4);
  int elements_count = shape.b * shape.w * shape.h * shape.d * aligned_channels;
  data.resize(elements_count * SizeOf(data_type));
  int32_t* gpu_data = reinterpret_cast<int32_t*>(data.data());
  DataToBHWDC(gpu_data, shape, *this, dst);
}

bool TensorDescriptor::SupportsZeroClamp(const Axis& axis) const {
  switch (storage_type) {
    case TensorStorageType::UNKNOWN:
      return false;
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return false;
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return axis == Axis::WIDTH || axis == Axis::HEIGHT;
    case TensorStorageType::TEXTURE_3D:
      return axis == Axis::WIDTH || axis == Axis::HEIGHT || axis == Axis::DEPTH;
  }
}

bool TensorDescriptor::CanReadOutOfBorder(const Axis& axis) const {
  switch (storage_type) {
    case TensorStorageType::UNKNOWN:
      return false;
    case TensorStorageType::BUFFER:
      return false;
    case TensorStorageType::IMAGE_BUFFER:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
    case TensorStorageType::TEXTURE_ARRAY:
      return true;
  }
}

bool TensorDescriptor::IsLinear() const {
  return storage_type == TensorStorageType::BUFFER ||
         storage_type == TensorStorageType::IMAGE_BUFFER;
}

bool TensorDescriptor::ReturnsZeroForNegOneRead() const {
  return storage_type == TensorStorageType::IMAGE_BUFFER;
}

absl::Status TensorDescriptor::CanCreateTensorWithShape(
    const GpuInfo& gpu_info, const BHWDC& shape) const {
  const int slices = DivideRoundUp(shape.c, 4);
  const uint64_t flt_size = data_type == DataType::FLOAT32 ? 4 : 2;
  const uint64_t channels = storage_type == TensorStorageType::SINGLE_TEXTURE_2D
                                ? shape.c
                                : slices * 4;
  const uint64_t allocation_size =
      flt_size * channels * shape.b * shape.w * shape.h * shape.d;
  const std::string common_desc = "Shape - " + ToString(shape) +
                                  ", data type - " + ToString(data_type) + ".";
  if (allocation_size > gpu_info.GetMaxMemoryAllocationSize()) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "Requested allocation size - ", allocation_size,
        " bytes. Max allocation size for this GPU - ",
        gpu_info.GetMaxMemoryAllocationSize(), " bytes. ", common_desc));
  }
  switch (storage_type) {
    case TensorStorageType::BUFFER: {
      const uint64_t flt4_size = 4 * (data_type == DataType::FLOAT32 ? 4 : 2);
      const uint64_t buffer_size =
          flt4_size * shape.b * shape.w * shape.h * shape.d * slices;
      if (buffer_size > gpu_info.GetMaxBufferSize()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Buffer with size - ", buffer_size,
            " bytes can not be created. Max buffer size for this GPU - ",
            gpu_info.GetMaxBufferSize(), " bytes. ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::IMAGE_BUFFER: {
      const uint64_t flt4_size = 4 * (data_type == DataType::FLOAT32 ? 4 : 2);
      const uint64_t buffer_size =
          flt4_size * shape.b * shape.w * shape.h * shape.d * slices;
      const uint64_t image_width = buffer_size / flt4_size;
      if (image_width > gpu_info.GetMaxImageBufferWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image buffer with width - ", image_width,
            " can not be created. Max image buffer width for this GPU - ",
            gpu_info.GetMaxImageBufferWidth(), ". ", common_desc));
      } else if (buffer_size > gpu_info.GetMaxBufferSize()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Buffer with size - ", buffer_size,
            " bytes can not be created. Max buffer size for this GPU - ",
            gpu_info.GetMaxBufferSize(), " bytes. ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::TEXTURE_3D: {
      if (gpu_info.IsApiOpenCl() &&
          gpu_info.opencl_info.cl_version < OpenClVersion::kCl1_2 &&
          slices == 1) {
        return absl::InternalError(
            "clCreateImage3D (that used in CL 1.0/1.1) can not create image "
            "with depth = 1 by specification.");
      }
      const int image_width = shape.w * shape.b;
      const int image_height = shape.h;
      const int image_depth = slices * shape.d;
      if (image_width > gpu_info.GetMaxImage3DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image3D with width - ", image_width,
            " can not be created. Max Image3D width for this GPU - ",
            gpu_info.GetMaxImage3DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage3DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image3D with height - ", image_height,
            " can not be created. Max Image3D height for this GPU - ",
            gpu_info.GetMaxImage3DHeight(), ". ", common_desc));
      } else if (image_depth > gpu_info.GetMaxImage3DDepth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image3D with depth - ", image_depth,
            " can not be created. Max Image3D depth for this GPU - ",
            gpu_info.GetMaxImage3DDepth(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::TEXTURE_ARRAY: {
      // Bug on some Adreno. b/131099086
      if (gpu_info.IsApiOpenCl() && slices == 1 && gpu_info.IsAdreno() &&
          !gpu_info.adreno_info.support_one_layer_texture_array) {
        return absl::InternalError(
            "Image2DArray with layer = 1 works incorrect on some Adreno in "
            "OpenCL. Can not be created.");
      }
      const int image_width = shape.w * shape.b;
      const int image_height = shape.h;
      const int image_layers = slices * shape.d;
      if (image_width > gpu_info.GetMaxImage2DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2DArray with width - ", image_width,
            " can not be created. Max Image2DArray width for this GPU - ",
            gpu_info.GetMaxImage2DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage2DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2DArray with height - ", image_height,
            " can not be created. Max Image2DArray height for this GPU - ",
            gpu_info.GetMaxImage2DHeight(), ". ", common_desc));
      } else if (image_layers > gpu_info.GetMaxImage2DArrayLayers()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2DArray with layers - ", image_layers,
            " can not be created. Max Image2DArray layers for this GPU - ",
            gpu_info.GetMaxImage2DArrayLayers(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::TEXTURE_2D: {
      const int image_width = shape.w * shape.b * shape.d;
      const int image_height = shape.h * slices;
      if (image_width > gpu_info.GetMaxImage2DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with width - ", image_width,
            " can not be created. Max Image2D width for this GPU - ",
            gpu_info.GetMaxImage2DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage2DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with height - ", image_height,
            " can not be created. Max Image2D height for this GPU - ",
            gpu_info.GetMaxImage2DHeight(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::SINGLE_TEXTURE_2D: {
      const int image_width = shape.w * shape.b * shape.d;
      const int image_height = shape.h;
      if (shape.c > 4) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with channels - ", shape.c, " can not be created."));
      } else if (!gpu_info.SupportsFloatImage2D(data_type, shape.c)) {
        return absl::ResourceExhaustedError(
            "Image2D doesn't support this pixel layout.");
      } else if (image_width > gpu_info.GetMaxImage2DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with width - ", image_width,
            " can not be created. Max Image2D width for this GPU - ",
            gpu_info.GetMaxImage2DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage2DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with height - ", image_height,
            " can not be created. Max Image2D height for this GPU - ",
            gpu_info.GetMaxImage2DHeight(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    default:
      return absl::UnimplementedError(
          "Can not create resources for unknown storage type.");
  }
}

absl::Status TensorDescriptor::CanCreateTensorWithShape(
    const GpuInfo& gpu_info, const BHWC& shape) const {
  const BHWDC shape5D(shape.b, shape.h, shape.w, 1, shape.c);
  return CanCreateTensorWithShape(gpu_info, shape5D);
}

namespace {
int GetLinearIndex(const TensorDescriptor& desc, const BHWDC& shape, int b,
                   int x, int y, int d, int s, int sub_c) {
  const int slices = DivideRoundUp(shape.c, 4);
  switch (desc.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return ((((d * slices + s) * shape.h + y) * shape.w + x) * shape.b + b) *
                 4 +
             sub_c;  // DSHWBC4
    case TensorStorageType::TEXTURE_2D:
      return ((((y * slices + s) * shape.w + x) * shape.b + b) * shape.d + d) *
                 4 +
             sub_c;  // HSWBDC4
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return (((y * shape.w + x) * shape.b + b) * shape.d + d) * shape.c +
             sub_c;  // HWBDC
    case TensorStorageType::UNKNOWN:
      return -1;
  }
}

int GetChannelsAlignment(const TensorDescriptor& desc, const BHWDC& shape) {
  return desc.storage_type == TensorStorageType::SINGLE_TEXTURE_2D ? shape.c
                                                                   : 4;
}
}  // namespace

template <typename FromType, typename ToType>
void DataFromBHWDC(const FromType* src, const BHWDC& shape,
                   const TensorDescriptor& desc, ToType* dst) {
  const int channels_alignment = GetChannelsAlignment(desc, shape);
  const int slices = DivideRoundUp(shape.c, 4);
  for (int b = 0; b < shape.b; ++b) {
    for (int s = 0; s < slices; ++s) {
      for (int y = 0; y < shape.h; ++y) {
        for (int x = 0; x < shape.w; ++x) {
          for (int d = 0; d < shape.d; ++d) {
            for (int c = 0; c < channels_alignment; ++c) {
              FromType value;
              if (s * 4 + c < shape.c) {
                const int cpu_index =
                    shape.LinearIndex({b, y, x, d, s * 4 + c});
                value = src[cpu_index];
              } else {
                value = 0;
              }
              int gpu_index = GetLinearIndex(desc, shape, b, x, y, d, s, c);
              dst[gpu_index] = value;
            }
          }
        }
      }
    }
  }
}

template void DataFromBHWDC<float, float>(const float* src, const BHWDC& shape,
                                          const TensorDescriptor& desc,
                                          float* dst);
template void DataFromBHWDC<float, half>(const float* src, const BHWDC& shape,
                                         const TensorDescriptor& desc,
                                         half* dst);
template void DataFromBHWDC<int32_t, int32_t>(const int32_t* src,
                                              const BHWDC& shape,
                                              const TensorDescriptor& desc,
                                              int32_t* dst);
template void DataFromBHWDC<int16_t, int16_t>(const int16_t* src,
                                              const BHWDC& shape,
                                              const TensorDescriptor& desc,
                                              int16_t* dst);
template void DataFromBHWDC<int8_t, int8_t>(const int8_t* src,
                                            const BHWDC& shape,
                                            const TensorDescriptor& desc,
                                            int8_t* dst);
template void DataFromBHWDC<uint32_t, uint32_t>(const uint32_t* src,
                                                const BHWDC& shape,
                                                const TensorDescriptor& desc,
                                                uint32_t* dst);
template void DataFromBHWDC<uint16_t, uint16_t>(const uint16_t* src,
                                                const BHWDC& shape,
                                                const TensorDescriptor& desc,
                                                uint16_t* dst);
template void DataFromBHWDC<uint8_t, uint8_t>(const uint8_t* src,
                                              const BHWDC& shape,
                                              const TensorDescriptor& desc,
                                              uint8_t* dst);

template <typename FromType, typename ToType>
void DataToBHWDC(const FromType* src, const BHWDC& shape,
                 const TensorDescriptor& desc, ToType* dst) {
  const int channels_alignment = GetChannelsAlignment(desc, shape);
  const int slices = DivideRoundUp(shape.c, 4);
  for (int b = 0; b < shape.b; ++b) {
    for (int s = 0; s < slices; ++s) {
      for (int y = 0; y < shape.h; ++y) {
        for (int x = 0; x < shape.w; ++x) {
          for (int d = 0; d < shape.d; ++d) {
            for (int c = 0; c < channels_alignment; ++c) {
              if (s * 4 + c >= shape.c) {
                continue;
              }
              int cpu_index = shape.LinearIndex({b, y, x, d, s * 4 + c});
              int gpu_index = GetLinearIndex(desc, shape, b, x, y, d, s, c);
              dst[cpu_index] = src[gpu_index];
            }
          }
        }
      }
    }
  }
}

template void DataToBHWDC<float, float>(const float* src, const BHWDC& shape,
                                        const TensorDescriptor& desc,
                                        float* dst);
template void DataToBHWDC<half, float>(const half* src, const BHWDC& shape,
                                       const TensorDescriptor& desc,
                                       float* dst);
template void DataToBHWDC<int32_t, int32_t>(const int32_t* src,
                                            const BHWDC& shape,
                                            const TensorDescriptor& desc,
                                            int32_t* dst);
template void DataToBHWDC<int16_t, int16_t>(const int16_t* src,
                                            const BHWDC& shape,
                                            const TensorDescriptor& desc,
                                            int16_t* dst);
template void DataToBHWDC<int8_t, int8_t>(const int8_t* src, const BHWDC& shape,
                                          const TensorDescriptor& desc,
                                          int8_t* dst);
template void DataToBHWDC<uint32_t, uint32_t>(const uint32_t* src,
                                              const BHWDC& shape,
                                              const TensorDescriptor& desc,
                                              uint32_t* dst);
template void DataToBHWDC<uint16_t, uint16_t>(const uint16_t* src,
                                              const BHWDC& shape,
                                              const TensorDescriptor& desc,
                                              uint16_t* dst);
template void DataToBHWDC<uint8_t, uint8_t>(const uint8_t* src,
                                            const BHWDC& shape,
                                            const TensorDescriptor& desc,
                                            uint8_t* dst);

}  // namespace gpu
}  // namespace tflite
