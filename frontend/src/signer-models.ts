/** The signers' files; the version is bumped with each rebuild of the avatars, so browsers do not keep a stale GLB. */
export const SIGNER_VERSION = "20261120";
export const SIGNER_MODELS = {
  male: `/models/tilmind-male.glb?v=${SIGNER_VERSION}`,
  female: `/models/tilmind-female.glb?v=${SIGNER_VERSION}`,
};
/**
 * The next avatars, for comparing in the words viewer: the face's shapes carried over from ICT-FaceKit's scans
 * (tools/ict_face.py) and the hoodie's sleeves drawn in (tools/tripo_sleeves.py).
 */
export const NEXT_SIGNER_MODELS = {
  male: `/models/next/tilmind-male.glb?v=${SIGNER_VERSION}`,
  female: `/models/next/tilmind-female.glb?v=${SIGNER_VERSION}`,
};
