#include "cylinder.h"
#include "../header/enum.h"

namespace rtx {
CylinderGeometry::CylinderGeometry(float radius, float height)
    : Geometry()
{
    _radius = radius;
    _y_max = height / 2.0f;
    _y_min = -height / 2.0f;
    _height = height;
    _transformation_matrix = glm::mat4(1.0f);
}
int CylinderGeometry::type() const
{
    return RTXGeometryTypeCylinder;
}
int CylinderGeometry::num_faces() const
{
    // parameter + transformation matrix + inverse of transformation matrix
    return 1 + 1 + 1;
}
int CylinderGeometry::num_vertices() const
{
    // radius + y_max + y_min + transformation matrix + inverse of transformation matrix
    return 1 + 3 + 3;
}
void CylinderGeometry::serialize_vertices(rtx::array<rtxVertex>& array, int offset) const
{
    // Add cylinder parameter
    array[0 + offset] = { _radius, _y_max, _y_min, -1.0f };

    // Add transformation matrix
    array[1 + offset] = { _transformation_matrix[0][0], _transformation_matrix[1][0], _transformation_matrix[2][0], _transformation_matrix[3][0] };
    array[2 + offset] = { _transformation_matrix[0][1], _transformation_matrix[1][1], _transformation_matrix[2][1], _transformation_matrix[3][1] };
    array[3 + offset] = { _transformation_matrix[0][2], _transformation_matrix[1][2], _transformation_matrix[2][2], _transformation_matrix[3][2] };

    // Add inverse transformation matrix
    glm::mat4 inv_transformation_matrix = glm::inverse(_transformation_matrix);
    array[4 + offset] = { inv_transformation_matrix[0][0], inv_transformation_matrix[1][0], inv_transformation_matrix[2][0], inv_transformation_matrix[3][0] };
    array[5 + offset] = { inv_transformation_matrix[0][1], inv_transformation_matrix[1][1], inv_transformation_matrix[2][1], inv_transformation_matrix[3][1] };
    array[6 + offset] = { inv_transformation_matrix[0][2], inv_transformation_matrix[1][2], inv_transformation_matrix[2][2], inv_transformation_matrix[3][2] };
}
void CylinderGeometry::serialize_faces(rtx::array<rtxFaceVertexIndex>& array, int offset) const
{
    // Cylinder parameter
    array[0 + offset] = { 0, -1, -1, -1 };

    // Transformation matrix
    array[1 + offset] = { 1, 2, 3, -1 };

    // Inverse transformation matrix
    array[2 + offset] = { 4, 5, 6, -1 };
}
void CylinderGeometry::set_transformation_matrix(glm::mat4f& transformation_matrix)
{
    _transformation_matrix = transformation_matrix;
}
std::shared_ptr<Geometry> CylinderGeometry::transoform(glm::mat4& transformation_matrix) const
{
    auto cylinder = std::make_shared<CylinderGeometry>(_radius, _height);
    cylinder->set_transformation_matrix(transformation_matrix);
    return cylinder;
}
}