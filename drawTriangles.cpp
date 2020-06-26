#include <student/gpu.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

bool isPointInside(Vec4 const*const p){
  // -Aw <= Ai <= +Aw
  for(int i=0;i<3;++i){
    if(p->data[i] <= -p->data[3])return false;
    if(p->data[i] >= +p->data[3])return false;
  }
  return true;
}

void PerspectiveDivision(Vec4*const ndc, Vec4 const*const pos){
  for(int a=0;a<3;++a)
    ndc->data[a] = pos->data[a]/pos->data[3];
    ndc->data[3] = pos->data[3];
}

Vec4 computeFragmentPositionTriangle(Vec4 const&p,uint32_t width,uint32_t height){
  Vec4 res;
  res.data[0] = (p.data[0]*.5f+.5f)*width; //x
  res.data[1] = (p.data[1]*.5f+.5f)*height; //y
  res.data[2] = p.data[2];
  res.data[3] = p.data[3];
  return res;
}

void copyVertexAttributeTriangle(GPU const*const gpu,GPUAttribute*const att,GPUVertexPullerHead const*const head,uint64_t vertexId){
  if(!head->enabled)return;
  GPUBuffer const*const buf = gpu_getBuffer(gpu,head->bufferId);
  uint8_t const*ptr = (uint8_t*)buf->data;
  uint32_t const offset = (uint32_t)head->offset;
  uint32_t const stride = (uint32_t)head->stride;
  uint32_t const size   = (uint32_t)head->type;
  memcpy(att->data,ptr+offset + vertexId*stride,size);
}

void vertexPuller(GPUInVertex*const inVertex, GPUVertexPuller const*const vao, GPU const*const gpu, uint32_t vertexSHaderInvocation){

  uint32_t vertexId; 
  
  if (gpu_isBuffer(gpu, vao->indices.bufferId))
  {
    GPUBuffer const*const buffer = gpu_getBuffer(gpu, vao->indices.bufferId);
    if (vao->indices.type == UINT8) 
    {
      vertexId = (uint32_t)((uint8_t*)(buffer->data))[vertexSHaderInvocation];
    }

    if (vao->indices.type == UINT16) 
    {
      vertexId = (uint32_t)((uint16_t*)(buffer->data))[vertexSHaderInvocation];
    }

    if (vao->indices.type == UINT32) 
    {
      vertexId = (uint32_t)((uint32_t*)(buffer->data))[vertexSHaderInvocation];
    }
  }
  else
  {
    vertexId = vertexSHaderInvocation;
  }
  
  inVertex->gl_VertexID = vertexId;

  for (int i = 0; i < MAX_ATTRIBUTES; i++)
  {
    copyVertexAttributeTriangle(gpu,inVertex->attributes+i,vao->heads+i,vertexId);
  }
  
}

/****************** Calculating Barycentric coordinates**********************/

void barycentric_weights(Vec2 v[3], Vec2 point, float w[3]) {
  float vector0 = v[1].data[1]-v[2].data[1];
  float vector1 = v[0].data[0]-v[2].data[0];
  float vector2 = v[2].data[0]-v[1].data[0];
  float vector3 = v[0].data[1]-v[2].data[1];
  float vector4 = v[2].data[1]-v[0].data[1];

  float divider = ((vector0)*(vector1)) +
                  ((vector2)*(vector3));

  w[0] = (((vector0)*(point.data[0]-v[2].data[0])) +
       ((vector2)*(point.data[1]-v[2].data[1]))) / divider;

  w[1] = (((vector4)*(point.data[0]-v[2].data[0])) +
       ((vector1)*(point.data[1]-v[2].data[1]))) / divider;

  w[2] = 1.0 - w[0] - w[1];
}

/***************************************************************************/

void triangle_Rasterization(GPUInFragment*const inFragment,GPU*const gpu,GPUOutVertex *vertex, GPUProgram const*const prg, GPUFragmentShaderData *fd){
  Vec4 pixels;
  Vec3 attribute_of_triangle;
  Vec3 etc;
  Vec2 pixel;
  Vec2 p[3];
  float w[3];
  float pixel_x;
  float pixel_y;
  float fragDepth;

  for(int i = 0; i < 3; i++){
    copy_Vec4_To_Vec2(&p[i],&vertex[i].gl_Position);
  }

  for (uint32_t x = 0; x < gpu->framebuffer.width; x++) {
    for (uint32_t y = 0; y < gpu->framebuffer.height; y++) {
      init_Vec2(&pixel,x+0.5,y+0.5);
      barycentric_weights(p,pixel,w);
      if ((w[0] >= 0) && (w[1] >= 0) && (w[2] >= 0)) {
        init_Vec3(&etc,vertex[0].gl_Position.data[3],vertex[1].gl_Position.data[3],vertex[2].gl_Position.data[3]);

        uint32_t iteration;
        enum AttributeType type;
        for (int a = 1; a < MAX_ATTRIBUTES; a++) {
          type  = prg->vs2fsType[a];

          if (type == ATTRIBUTE_EMPTY) {
          	continue;
          }
          else if (type == ATTRIBUTE_FLOAT) {
            iteration = 1;
              float frag,vc;
              for (uint8_t i = 0; i < 3; i++) {
                memcpy(&vc,vertex[i].attributes[a].data,sizeof(float)*iteration);
                frag += (w[i])*(vc);
            }

          }

          else if (type == ATTRIBUTE_VEC2 ) {
            iteration = 2;
            Vec2 out,frag,vc;
            for (uint8_t i = 0; i < 3; i++) {
              memcpy(&vc,vertex[i].attributes[a].data,sizeof(float)*iteration);
              multiply_Vec2_Float(&out,&vc,w[i]);
              add_Vec2(&frag,&out,&frag);
            }

          }
          else if  (type == ATTRIBUTE_VEC3) {
            iteration = 3;
            Vec3 out,frag,vc;
            for (uint8_t i = 0; i < 3; i++) {
              memcpy(&vc,vertex[i].attributes[a].data,sizeof(float)*iteration);
              multiply_Vec3_Float(&out,&vc,w[i]);
              add_Vec3(&frag,&out,&frag);
            }

          }
          else if (type == ATTRIBUTE_VEC4) {
            iteration = 4;
            Vec4 out,frag,vc;
            for (uint8_t i = 0; i < 3; i++) {
              memcpy(&vc,vertex[i].attributes[a].data,sizeof(float)*iteration);
              multiply_Vec4_Float(&out,&vc,w[i]);
              add_Vec4(&frag,&out,&frag);
            }

          }

          memcpy(inFragment->attributes[a].data,&pixels.data,sizeof(float)*iteration);
        }
        prg->fragmentShader(fd);
        perFragmentOperationTriangle(&fd->outFragment,gpu,inFragment->gl_FragCoord);
      }
    }
  }
}

void perFragmentOperationTriangle(GPUOutFragment const*const outFragment,GPU*const gpu,Vec4 ndc){
  Vec4 coord = ndc;//computeFragmentPositionTriangle(ndc,gpu->framebuffer.width,gpu->framebuffer.height);
  GPUFramebuffer*const frame = &gpu->framebuffer;

  if(coord.data[0] < 0 || coord.data[0] >= frame->width)return;
  if(coord.data[1] < 0 || coord.data[1] >= frame->height)return;
  if(isnan(coord.data[0]))return;
  if(isnan(coord.data[1]))return;
  uint32_t const pixCoord = frame->width*(int)coord.data[1]+(int)coord.data[0];
  Vec4 color;
  PerspectiveDivision(&color, &outFragment->gl_FragColor);
  frame->color[pixCoord] = color;
  frame->depth[pixCoord] = coord.data[2];

}
/**
 * @brief This function should draw triangles
 *
 * @param gpu gpu
 * @param nofVertices number of vertices
 */
void gpu_drawTriangles(GPU *const gpu, uint32_t nofVertices)
{

  /// \todo Naimplementujte vykreslování trojúhelníků.
  /// nofVertices - počet vrcholů
  /// gpu - data na grafické kartě
  /// Vašim úkolem je naimplementovat chování grafické karty.
  /// Úkol je složen:
  /// 1. z implementace Vertex Pulleru
  /// 2. zavolání vertex shaderu pro každý vrchol
  /// 3. rasterizace
  /// 4. zavolání fragment shaderu pro každý fragment
  /// 5. zavolání per fragment operací nad fragmenty (depth test, zápis barvy a hloubky)
  /// Více v připojeném videu.
  GPUProgram      const* prg = gpu_getActiveProgram(gpu);
  GPUVertexPuller const* vao = gpu_getActivePuller (gpu);

  GPUVertexShaderData   vd; 
  GPUFragmentShaderData fd; //GPUinFragment, GPUoutFragment

  vd.uniforms = &prg->uniforms;
  fd.uniforms = &prg->uniforms;

  GPUOutVertex vertex[nofVertices];
  Vec3 etc;
  Vec4 pos,ndc,vprt;
  for(uint32_t v = 0; v<nofVertices; ++v){
    vertexPuller(&vd.inVertex,vao,gpu,v);

    prg->vertexShader(&vd);

    copy_Vec4(&pos,&vd.outVertex.gl_Position);

      PerspectiveDivision(&ndc,&pos);

    vprt = computeFragmentPositionTriangle(ndc,gpu->framebuffer.width,gpu->framebuffer.height);
    copy_Vec4(&vd.outVertex.gl_Position,&vprt);
    memcpy(&vertex[v],&vd.outVertex,sizeof(struct GPUOutVertex));
  }

  triangle_Rasterization(&fd.inFragment,gpu,vertex,prg,&fd);

  (void)gpu;
  (void)nofVertices; 
}

